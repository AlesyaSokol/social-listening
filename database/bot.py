import requests


bot_token = '8026384357:AAFmSwI4j8d5V41D4QeHqN4hS1ff9sk3V98'

def send_update_tg(text="Тестовый текст"):
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    data = {'chat_id': '-1002745156752', 'text': text, 'parse_mode': 'Markdown'}
    r = requests.post(url, data=data)

if __name__ == '__main__':
    send_update_tg()

