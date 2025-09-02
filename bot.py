from collections import defaultdict
from datetime import datetime

import nextcord
from google import genai
from google.genai.types import GenerateContentConfig
from nextcord.ext import commands


description = "my bot"
intents = nextcord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", description=description, intents=intents)
client = genai.Client()
message_history = defaultdict(list)
chat_sessions = {}


with open("system.txt", "r", encoding="utf-8") as file1:
    with open("all_stories.txt", "r", encoding="utf-8") as file:
        base_system_instruction = f"""{file1.read()}\n\n{file.read()}"""


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # don't talk to itself
    user_id = str(message.author.id)
    message_history[user_id].append(
        {"content": message.content, "timestamp": message.created_at, "is_bot": False}
    )
    if len(message_history[user_id]) > 10:
        message_history[user_id].pop(0)  # Remove oldest message
    if bot.user in message.mentions:
        print(f"we got a message from {message.author.name}!")
        try:
            if user_id not in chat_sessions:
                chat_sessions[user_id] = client.chats.create(
                    model="gemini-2.0-flash",
                    config=GenerateContentConfig(
                        system_instruction=base_system_instruction,
                    ),
                )
            chat = chat_sessions[user_id]
            response = chat.send_message(message.content)
            await message.channel.send(response.text.strip())
            message_history[user_id].append(
                {
                    "content": response.text.strip(),
                    "timestamp": datetime.now(),
                    "is_bot": True,
                }
            )
            if len(message_history[user_id]) > 10:
                message_history[user_id].pop(0)  # Maintain limit

        except Exception as e:
            await message.channel.send(f"Error generating response: {e}")
    await bot.process_commands(message)  # Keep command handling alive


with open("TOKEN.txt", "r", encoding="utf-8") as file:
    TOKEN = file.read().strip()
print("Starting bot...")
bot.run(TOKEN)
