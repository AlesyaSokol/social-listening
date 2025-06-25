import requests
import dotenv
import os

dotenv.load_dotenv('.env')

BOT_TOKEN = os.getenv('BOT_TOKEN')  
CHAT_ID = os.getenv('CHAT_ID')  

def send_update_tg(text="Тестовый текст"):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    data = {'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown', 'disable_web_page_preview': True}
    r = requests.post(url, data=data)

if __name__ == '__main__':
    send_update_tg()

