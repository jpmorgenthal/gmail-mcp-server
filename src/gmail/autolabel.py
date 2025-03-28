import venv
from typing import Any
import argparse
import os
import asyncio
import logging
import base64
import json
from email.message import EmailMessage
from email.header import decode_header
from base64 import urlsafe_b64decode
from email import message_from_bytes
import webbrowser

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio


from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMAIL_ADMIN_PROMPTS = """You are an email administrator. 
You can draft, edit, read, trash, open, and send emails.
You've been given access to a specific gmail account. 
You have the following tools available:
- Send an email (send-email)
- Retrieve unread emails (get-unread-emails)
- Read email content (read-email)
- Trash email (trash-email)
- Open email in browser (open-email)
Never send an email draft or trash an email unless the user confirms first. 
Always ask for approval if not already given.
"""

# Define available prompts
PROMPTS = {
    "manage-email": types.Prompt(
        name="manage-email",
        description="Act like an email administator",
        arguments=None,
    ),
    "draft-email": types.Prompt(
        name="draft-email",
        description="Draft an email with content and recipient",
        arguments=[
            types.PromptArgument(
                name="content",
                description="What the email is about",
                required=True
            ),
            types.PromptArgument(
                name="recipient",
                description="Who should the email be addressed to",
                required=True
            ),
            types.PromptArgument(
                name="recipient_email",
                description="Recipient's email address",
                required=True
            ),
        ],
    ),
    "edit-draft": types.Prompt(
        name="edit-draft",
        description="Edit the existing email draft",
        arguments=[
            types.PromptArgument(
                name="changes",
                description="What changes should be made to the draft",
                required=True
            ),
            types.PromptArgument(
                name="current_draft",
                description="The current draft to edit",
                required=True
            ),
        ],
    ),
}


def decode_mime_header(header: str) -> str: 
    """Helper function to decode encoded email headers"""
    
    decoded_parts = decode_header(header)
    decoded_string = ''
    for part, encoding in decoded_parts: 
        if isinstance(part, bytes): 
            # Decode bytes to string using the specified encoding 
            decoded_string += part.decode(encoding or 'utf-8') 
        else: 
            # Already a string 
            decoded_string += part 
    return decoded_string


class GmailService:
    def __init__(self,
                 creds_file_path: str,
                 token_path: str,
                 scopes: list[str] = ['https://www.googleapis.com/auth/gmail.modify']):
        logger.info(f"Initializing GmailService with creds file: {creds_file_path}")
        self.creds_file_path = creds_file_path
        self.token_path = token_path
        self.scopes = scopes
        self.token = self._get_token()
        logger.info("Token retrieved successfully")
        self.service = self._get_service()
        logger.info("Gmail service initialized")
        self.user_email = self._get_user_email()
        logger.info(f"User email retrieved: {self.user_email}")

    def load_user_secrets_from_local(self, user_secrets_file, scopes):
        logger.info(f'Loading user secrets from {user_secrets_file}')
        with open(user_secrets_file, 'r') as stream:
            creds_json = json.load(stream)
            creds = Credentials.from_authorized_user_info(creds_json, scopes)
            # workaround for
            # https://github.com/googleapis/google-auth-library-python/issues/501
            creds.token = creds_json['token']
            return creds
        return None

    def _get_token(self) -> Credentials:
        """Get or refresh Google API token"""

        token = None
    
        if os.path.exists(self.token_path):
            logger.info('Loading token from file')
            token = self.load_user_secrets_from_local(self.token_path, self.scopes)

        if not token or not token.valid:
            if token and token.expired and token.refresh_token:
                logger.info('Refreshing token')
                token.refresh(Request())
            else:
                logger.info('Fetching new token')
                flow = InstalledAppFlow.from_client_secrets_file(self.creds_file_path, self.scopes)
                token = flow.run_local_server()

            with open(self.token_path, 'w') as token_file:
                token_file.write(token.to_json())
                logger.info(f'Token saved to {self.token_path}')

        return token

    def _get_service(self) -> Any:
        """Initialize Gmail API service"""
        try:
            service = build('gmail', 'v1', credentials=self.token)
            return service
        except HttpError as error:
            logger.error(f'An error occurred building Gmail service: {error}')
            raise ValueError(f'An error occurred: {error}')
    
    def _get_user_email(self) -> str:
        """Get user email address"""
        profile = self.service.users().getProfile(userId='me').execute()
        user_email = profile.get('emailAddress', '')
        return user_email
    
    async def send_email(self, recipient_id: str, subject: str, message: str,) -> dict:
        """Creates and sends an email message"""
        try:
            message_obj = EmailMessage()
            message_obj.set_content(message)
            
            message_obj['To'] = recipient_id
            message_obj['From'] = self.user_email
            message_obj['Subject'] = subject

            encoded_message = base64.urlsafe_b64encode(message_obj.as_bytes()).decode()
            create_message = {'raw': encoded_message}
            
            send_message = await asyncio.to_thread(
                self.service.users().messages().send(userId="me", body=create_message).execute
            )
            logger.info(f"Message sent: {send_message['id']}")
            return {"status": "success", "message_id": send_message["id"]}
        except HttpError as error:
            return {"status": "error", "error_message": str(error)}

    async def open_email(self, email_id: str) -> str:
        """Opens email in browser given ID."""
        try:
            url = f"https://mail.google.com/#all/{email_id}"
            webbrowser.open(url, new=0, autoraise=True)
            return "Email opened in browser successfully."
        except HttpError as error:
            return f"An HttpError occurred: {str(error)}"

    async def get_unread_emails(self) -> list[dict[str, str]]| str:
        """
        Retrieves unread messages from mailbox.
        Returns list of messsage IDs in key 'id'."""
        try:
            user_id = 'me'
            query = 'in:inbox is:unread'

            response = self.service.users().messages().list(userId=user_id,
                                                        q=query,
                                                        maxResults=20).execute()
            messages = []
            if 'messages' in response:
                messages.extend(response['messages'])

#            while 'nextPageToken' in response:
#                page_token = response['nextPageToken']
#                response = self.service.users().messages().list(userId=user_id, q=query,
#                                                    pageToken=page_token).execute()
#                messages.extend(response['messages'])
            return messages

        except HttpError as error:
            return f"An HttpError occurred: {str(error)}"


    async def label_email(self, email_id:str, label:str) -> str:
        """
        Retrieves unread messages from mailbox.
        Returns list of messsage IDs in key 'id'."""
        try:
            user_id = 'me'
            logger.info("Entered label_email")
            labelList = self.service.users().labels().list(userId=user_id).execute()

            if 'labels' not in labelList:
                return "No labels found in the account."
                
            labelID = next((labeli['id'] for labeli in labelList['labels'] if labeli.get('name') == label), None)
            logger.info(f"Label ID: {labelID}")
            if not labelID:
                return json.dumps({'error': 'true', 'message': f"Label '{label}' not found."})

            message = self.service.users().messages().modify(userId=user_id, 
                                        id=email_id, 
                                        body={'addLabelIds': [labelID]}).execute()
            logger.info(f"Modify Results:{message}")
            return json.dumps(message)

        except HttpError as error:
            return f"An ttpError occurred: {str(error)}"
        
    def count_words(self, text: str) -> int:
        """Returns the number of words in a string."""
        if not text:
            return 0
        return len(text.split())
    
    async def read_email(self, email_id: str) -> dict[str, str]| str:
        """Retrieves email contents including to, from, subject, and contents."""
        try:
            msg = self.service.messages().get(userId="me", id=email_id, format='raw').execute()
            email_metadata = {}

            # Decode the base64URL encoded raw content
            raw_data = msg['raw']
            decoded_data = urlsafe_b64decode(raw_data)
#            logger.info(f"Decoded data: {decoded_data}")

            # Parse the RFC 2822 email
            mime_message = message_from_bytes(decoded_data)
#            logger.info(f"Mime message: {mime_message}")
            # Extract the email body
            body = None
            if mime_message.is_multipart():
                for part in mime_message.walk():
                    # Extract the text/plain part
                    logger.info(f"Part: {part.get_content_type()}")
                    if part.get_content_type() == "text/plain":
                        try:
                            body = mime_message.get_payload(decode=True).decode('utf-8')
                        except UnicodeDecodeError:
                            # Fallback to other common encodings or ignore errors
                            body = mime_message.get_payload(decode=True).decode('latin-1', errors='replace')
                    break
            else:
                # For non-multipart messages
                body = mime_message.get_payload(decode=True).decode('utf-8')

            if not body:
                # Fallback to the raw content if no text/plain part is found
                body = "THIS IS SPAM"

            if self.count_words(body) > 10000:
                # Truncate the body to the first 10000 words 
                words = body.split()
                body = ' '.join(words[:10000])

            email_metadata['content'] = body
            
            # Extract metadata
            email_metadata['subject'] = decode_mime_header(mime_message.get('subject', ''))
            email_metadata['from'] = mime_message.get('from','')
            email_metadata['to'] = mime_message.get('to','')
            email_metadata['date'] = mime_message.get('date','')
            
            logger.info(f"Email read: {email_id}")
            
            # We want to mark email as read once we read it
            await self.mark_email_as_read(email_id)

            return email_metadata
        except HttpError as error:
            return f"An HttpError occurred: {str(error)}"
        
    async def trash_email(self, email_id: str) -> str:
        """Moves email to trash given ID."""
        try:
            self.service.users().messages().trash(userId="me", id=email_id).execute()
            logger.info(f"Email moved to trash: {email_id}")
            return "Email moved to trash successfully."
        except HttpError as error:
            return f"An HttpError occurred: {str(error)}"
        
    async def mark_email_as_read(self, email_id: str) -> str:
        """Marks email as read given ID."""
        try:
            self.service.users().messages().modify(userId="me", id=email_id, body={'removeLabelIds': ['UNREAD']}).execute()
            logger.info(f"Email marked as read: {email_id}")
            return "Email marked as read."
        except HttpError as error:
            return f"An HttpError occurred: {str(error)}"
  
    async def process_and_label_emails(self) -> list[dict]:
        """
        Processes unread emails, determines their importance using Ollama, and labels unimportant emails as 'Ads'.

        Args:
            gmail_service (GmailService): An instance of the GmailService class.
            ollama_api_url (str): The API endpoint for Ollama.

        Returns:
            list[dict]: A list of results containing email metadata and actions taken.
        """
        ollama_api_url = "http://localhost:11434/api/generate"
        try:
            # Step 1: Retrieve unread emails
            unread_emails = await self.get_unread_emails()
            if not unread_emails:
                logger.info("No unread emails found.")
                return []

            results = []

            # Step 2: Iterate over email IDs and process each email
            for email in unread_emails:
                email_id = email.get("id")
                if not email_id:
                    continue

                # Step 3: Retrieve email content
                email_content = await self.read_email(email_id)
                if not email_content:
                    logger.info(f"Email {email_id} not found or already processed.")
                    continue
                if isinstance(email_content, str):  # Handle error case
                    logger.error(f"Failed to read email {email_id}: {email_content}")
                    continue

                # Step 4: Send email content to Ollama for importance determination
                response = await self.send_to_ollama(ollama_api_url, email_content)
                
                logger.debug(f"Ollama response for email {email_id}: {response}")
            
                label = response.get("message", False).get("content", False)
                # Remove entire string containing <think> tags from labels
                #if label and "</think>" in label:
                    #label = label.split("</think>")[1].strip()
                logger.info(f"Label determined for email {email_id}: {label}")

                # Step 5: Label unimportant emails as 'Ads'
                if label:
                    label_response = await self.label_email(email_id, label)

                # Step 6: Append results
                results.append({
                    "email_id": email_id,
                    "label": label,
                    "label_response": label_response,
                    "ollama_response": response
                })

            return results

        except Exception as e:
            logger.error(f"An error occurred while processing emails: {str(e)}")
            return []

    async def send_to_ollama(self, api_url: str, email_content: dict) -> dict:
        """
        Sends email content to Ollama API to determine if it is important.

        Args:
            api_url (str): The API endpoint for Ollama.
            email_content (dict): The email content to analyze.

        Returns:
            dict: The response from Ollama.
        """
        import aiohttp

        try:
            async with aiohttp.ClientSession(headers={'Content-Type': 'application/json'}) as session:
                payload = {
                    "model": "llama3.2",
                    "prompt": "You are an email assistant. "
                                "Determine if the following email is an advertisement, spam or important. "
                                "if important, respond with 'Review' If an advertisment respond with 'Ads' If it discusses political themes respond with 'TRASH' "
                                "if email does not identify me directly as JP or Jeffrey then consider it an advertisement. "
                                "If the sentiment is personal, respond with 'Review' " 
                                "If the content is HTML don't evaluate it instead use the reply-to address to determine if it is spam or not. "
                                "return only one word answers." +
                                json.dumps(email_content),
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "max_tokens": 100,
                    },
                }
                #logger.info(payload)
                
                logger.debug(f"Sending request to Oa API: {payload}")
                headers = {'Content-Type': 'application/json'}
                async with session.post(api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        logger.debug("Received response from Ollama API")
                        return await response.json()
                    else:
                        logger.error(f"Ollama API returned status {response}")
                        return {"is_important": False, "error": f"Status {response.status}"}
        except Exception as e:
            logger.error(f"An error occurred while sending data to Ollama: {str(e)}")
            return {"is_important": False, "error": str(e)}
    
async def main(creds_file_path: str,
               token_path: str):
    
    gmail_service = GmailService(creds_file_path, token_path)
    server = Server("gmail")

    @server.list_prompts()
    async def list_prompts() -> list[types.Prompt]:
        return list(PROMPTS.values())

    @server.get_prompt()
    async def get_prompt(
        name: str, arguments: dict[str, str] | None = None
    ) -> types.GetPromptResult:
        if name not in PROMPTS:
            raise ValueError(f"Prompt not found: {name}")

        if name == "manage-email":
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=EMAIL_ADMIN_PROMPTS,
                        )
                    )
                ]
            )

        if name == "draft-email":
            content = arguments.get("content", "")
            recipient = arguments.get("recipient", "")
            recipient_email = arguments.get("recipient_email", "")
            
            # First message asks the LLM to create the draft
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"""Please draft an email about {content} for {recipient} ({recipient_email}).
                            Include a subject line starting with 'Subject:' on the first line.
                            Do not send the email yet, just draft it and ask the user for their thoughts."""
                        )
                    )
                ]
            )
        
        elif name == "edit-draft":
            changes = arguments.get("changes", "")
            current_draft = arguments.get("current_draft", "")
            
            # Edit existing draft based on requested changes
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"""Please revise the current email draft:
                            {current_draft}
                            
                            Requested changes:
                            {changes}
                            
                            Please provide the updated draft."""
                        )
                    )
                ]
            )

        raise ValueError("Prompt implementation not found")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="send-email",
                description="""Sends email to recipient. 
                Do not use if user only asked to draft email. 
                Drafts must be approved before sending.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "recipient_id": {
                            "type": "string",
                            "description": "Recipient email address",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject",
                        },
                        "message": {
                            "type": "string",
                            "description": "Email content text",
                        },
                    },
                    "required": ["recipient_id", "subject", "message"],
                },
            ),
            types.Tool(
                name="trash-email",
                description="""Moves email to trash. 
                Confirm before moving email to trash.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "Email ID",
                        },
                    },
                    "required": ["email_id"],
                },
            ),
            types.Tool(
                name="get-unread-emails",
                description="Retrieve unread emails",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
            ),
            types.Tool(
                name="label-email",
                description="Label email with given label",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "Email ID",
                        },
                        "label": {
                            "type": "string",
                            "description": "Label to apply",
                        },
                    },
                    "required": ["email_id", "label"],
                },
            ),
            types.Tool(
                name="read-email",
                description="Retrieves given email content",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "Email ID",
                        },
                    },
                    "required": ["email_id"],
                },
            ),
            types.Tool(
                name="mark-email-as-read",
                description="Marks given email as read",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "Email ID",
                        },
                    },
                    "required": ["email_id"],
                },
            ),
            types.Tool(
                name="open-email",
                description="Open email in browser",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "Email ID",
                        },
                    },
                    "required": ["email_id"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:

        if name == "label-email":
            logger.info("Labeling email")
            label = arguments.get("label")
            if not label:
                raise ValueError("Missing recipient parameter")
            logger.info(f"Label: {label}")
            email_id = arguments.get("email_id")
            if not email_id:
                raise ValueError("Missing email ID parameter")
            logger.info(f"Email ID: {email_id}")
            send_response = await gmail_service.label_email(email_id, label)
            json_data = json.loads(send_response)
            logger.info(f"Labeling response: {send_response}")
            if not json_data.get("id", None):
                # If the label was not found, return an error message
                raise ValueError(f"{json_data['message']}")
            
            response_text = f"Email successfully classified {email_id}"
            return [types.TextContent(type="text", text=response_text)]

        if name == "send-email":
            recipient = arguments.get("recipient_id")
            if not recipient:
                raise ValueError("Missing recipient parameter")
            subject = arguments.get("subject")
            if not subject:
                raise ValueError("Missing subject parameter")
            message = arguments.get("message")
            if not message:
                raise ValueError("Missing message parameter")
                
            # Extract subject and message content
            email_lines = message.split('\n')
            if email_lines[0].startswith('Subject:'):
                subject = email_lines[0][8:].strip()
                message_content = '\n'.join(email_lines[1:]).strip()
            else:
                message_content = message
                
            send_response = await gmail_service.send_email(recipient, subject, message_content)
            
            if send_response["status"] == "success":
                response_text = f"Email sent successfully. Message ID: {send_response['message_id']}"
            else:
                response_text = f"Failed to send email: {send_response['error_message']}"
            return [types.TextContent(type="text", text=response_text)]

        if name == "get-unread-emails":
                
            unread_emails = await gmail_service.get_unread_emails()
            return [types.TextContent(type="text", text=str(unread_emails),artifact={"type": "json", "data": unread_emails} )]
        
        if name == "read-email":
            email_id = arguments.get("email_id")
            if not email_id:
                raise ValueError("Missing email ID parameter")
                
            retrieved_email = await gmail_service.read_email(email_id)
            return [types.TextContent(type="text", text=str(retrieved_email),artifact={"type": "dictionary", "data": retrieved_email} )]
        if name == "open-email":
            email_id = arguments.get("email_id")
            if not email_id:
                raise ValueError("Missing email ID parameter")
                
            msg = await gmail_service.open_email(email_id)
            return [types.TextContent(type="text", text=str(msg))]
        if name == "trash-email":
            email_id = arguments.get("email_id")
            if not email_id:
                raise ValueError("Missing email ID parameter")
                
            msg = await gmail_service.trash_email(email_id)
            return [types.TextContent(type="text", text=str(msg))]
        if name == "mark-email-as-read":
            email_id = arguments.get("email_id")
            if not email_id:
                raise ValueError("Missing email ID parameter")
                
            msg = await gmail_service.mark_email_as_read(email_id)
            return [types.TextContent(type="text", text=str(msg))]
        else:
            logger.error(f"Unknown tool: {name}")
            raise ValueError(f"Unknown tool: {name}")

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="gmail",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

async def explicit(creds_file_path: str,
               token_path: str):
    
    gmail_service = GmailService(creds_file_path, token_path)
    server = Server("gmail")
    results = await gmail_service.process_and_label_emails()
    logger.info(f"Results: {results}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gmail API MCP Server')
    parser.add_argument('--creds-file-path',
                        required=True,
                       help='OAuth 2.0 credentials file path')
    parser.add_argument('--token-path',
                        required=True,
                       help='File location to store and retrieve access and refresh tokens for application')
    
    args = parser.parse_args()
#    asyncio.run(main(args.creds_file_path, args.token_path))
    asyncio.run(explicit(args.creds_file_path, args.token_path))
    
