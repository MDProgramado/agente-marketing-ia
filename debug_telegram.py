import asyncio
import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    telegram_bot_token: str
    telegram_chat_id: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

async def debug_bot():
    print(f"--- DIAGNÓSTICO DO TELEGRAM ---")
    print(f"Token sendo usado: {settings.telegram_bot_token[:10]}...")
    
    async with httpx.AsyncClient() as client:
        # 1. Quem sou eu? (Verifica se o Token existe)
        url_me = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe"
        resp_me = await client.get(url_me)
        print(f"\n1. Status do Bot: {resp_me.status_code}")
        print(f"   Detalhes: {resp_me.json()}")

        # 2. Quem falou comigo? (Tenta pegar seu ID real se vc mandou 'Oi')
        url_updates = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"
        resp_updates = await client.get(url_updates)
        updates = resp_updates.json()
        
        print(f"\n2. Últimas mensagens recebidas:")
        if updates.get("result"):
            last_msg = updates["result"][-1]
            chat_id = last_msg["message"]["chat"]["id"]
            user_name = last_msg["message"]["from"].get("first_name", "Desconhecido")
            print(f"   ✅ MENSAGEM ENCONTRADA de {user_name}!")
            print(f"   🚨 SEU CHAT ID CORRETO É: {chat_id}")
            print(f"   (Compare com o do .env: {settings.telegram_chat_id})")
            
            # 3. Teste de Envio Forçado
            print(f"\n3. Tentando enviar mensagem de teste para {chat_id}...")
            url_send = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            resp_send = await client.post(url_send, json={
                "chat_id": chat_id, 
                "text": "🤖 Teste de Diagnóstico: Se você ler isso, funcionou!"
            })
            print(f"   Status do Envio: {resp_send.status_code}")
            print(f"   Resposta do Telegram: {resp_send.json()}")
            
        else:
            print("   ⚠️ Nenhuma mensagem encontrada. VOCÊ MANDOU 'Oi' PARA O BOT?")
            print("   Vá no Telegram, ache o bot e mande uma mensagem agora.")

if __name__ == "__main__":
    asyncio.run(debug_bot())