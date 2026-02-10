import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional

import httpx
import instructor
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlmodel import Field as DBField, SQLModel, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 1. CONFIGURAÇÃO & SEGURANÇA ---

class Settings(BaseSettings):
    serper_api_key: str
    openrouter_api_key: str  
    # openai_api_key removida para evitar erro de validação
    telegram_bot_token: str
    telegram_chat_id: str
    database_url: str = "sqlite+aiosqlite:///marketing_agent.db"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

engine = create_async_engine(settings.database_url, echo=False)

# --- 2. CAMADA DE DADOS (Persistência) ---

class SentProduct(SQLModel, table=True):
    id: Optional[int] = DBField(default=None, primary_key=True)
    product_name: str = DBField(index=True)
    virality_score: int
    # Correção do warning de datetime: Usando UTC explicitamente
    sent_at: datetime = DBField(default_factory=lambda: datetime.now(timezone.utc))

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

# --- 3. MODELAGEM SEMÂNTICA ---

class ProductOpportunity(BaseModel):
    product_name: str = Field(..., description="Nome comercial curto do produto.")
    virality_score: int = Field(..., ge=0, le=100)
    target_audience: str
    pain_point_solved: str
    visual_prompt: str = Field(..., description="Prompt detalhado em INGLÊS para DALL-E 3.")
    marketing_hook: str
    hashtags: List[str]

class MarketAnalysisResult(BaseModel):
    top_opportunities: List[ProductOpportunity]
    market_mood: Literal["Alta Demanda", "Saturado", "Emergente", "Estável"]
    strategy_advice: str

# --- 4. O AGENTE SUPERIOR ---

class MarketingAgent:
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        self.reasoning_client = instructor.from_openai(
            AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            ),
            mode=instructor.Mode.JSON
        )
        
        # Cliente Visual usando OpenRouter
        self.visual_client = AsyncOpenAI(
            api_key=settings.openrouter_api_key, 
            base_url="https://openrouter.ai/api/v1"
        ) 

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_trends(self, query: str) -> dict:
        logger.info(f"🔍 [COLETA] Sondando Google Trends: {query}")
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "gl": "br", "hl": "pt-br", "num": 8, "tbs": "qdr:w"})
        headers = {'X-API-KEY': settings.serper_api_key, 'Content-Type': 'application/json'}
        
        response = await self.http_client.post(url, headers=headers, content=payload)
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def analyze_market(self, search_data: dict) -> MarketAnalysisResult:
        logger.info("🧠 [ANÁLISE] Raciocinando sobre os dados...")
        organic = search_data.get("organic", [])
        if not organic:
            return MarketAnalysisResult(top_opportunities=[], market_mood="Estável", strategy_advice="Sem dados.")

        snippets = "\n".join([f"- {item.get('title')}: {item.get('snippet')}" for item in organic])
        
        return await self.reasoning_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            response_model=MarketAnalysisResult,
            messages=[
                {"role": "system", "content": "Você é um Diretor de Marketing Viral. Identifique produtos físicos."},
                {"role": "user", "content": f"Analise estes resultados de busca e encontre produtos vencedores:\n{snippets}"}
            ],
        )

   # ... (resto do código acima)

    async def generate_viral_image(self, prompt: str) -> Optional[str]:
        """
        Gera imagem usando Pollinations AI (Gratuito/Rápido) 
        para contornar limitações de imagem da OpenRouter.
        """
        logger.info(f"🎨 [ARTE] Gerando imagem via Pollinations: {prompt[:30]}...")
        try:
            # Truque de Engenharia: Codificamos o prompt na URL
            # Isso gera uma imagem na hora, sem precisar de API Key complexa
            import urllib.parse
            encoded_prompt = urllib.parse.quote(f"product photography, 8k, cinematic lighting, {prompt}")
            
            # Retorna a URL direta. O Telegram vai baixar e mostrar automaticamente.
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
            
            return image_url
        except Exception as e:
            logger.error(f"⚠️ Falha na geração de imagem: {e}")
            return None

    # ... (resto do código abaixo)
    async def check_memory(self, product_name: str) -> bool:
        """Verifica se o produto já foi enviado."""
        async with AsyncSession(engine) as session:
            # Correção aqui: Usando execute() e scalars()
            statement = select(SentProduct).where(
                SentProduct.product_name == product_name,
                SentProduct.sent_at > datetime.now(timezone.utc) - timedelta(days=7)
            )
            result = await session.execute(statement)
            existing = result.scalars().first()
            
            if existing:
                logger.info(f"🚫 [MEMÓRIA] '{product_name}' já foi recomendado recentemente.")
                return True
            return False

    async def save_to_memory(self, product: ProductOpportunity):
        async with AsyncSession(engine) as session:
            entry = SentProduct(
                product_name=product.product_name,
                virality_score=product.virality_score,
                sent_at=datetime.now(timezone.utc)
            )
            session.add(entry)
            await session.commit()

    async def send_alert(self, product: ProductOpportunity, image_url: str = None):
        base_url = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
        caption = (
            f"🔥 **{product.product_name.upper()}** (Score: {product.virality_score})\n\n"
            f"🎯 **Público:** {product.target_audience}\n"
            f"💡 **Estratégia:** {product.marketing_hook}\n"
            f"⚠️ **Resolve:** {product.pain_point_solved}\n\n"
            f"🏷️ `{' '.join(product.hashtags[:5])}`"
        )
        try:
            if image_url:
                await self.http_client.post(
                    f"{base_url}/sendPhoto",
                    json={"chat_id": settings.telegram_chat_id, "photo": image_url, "caption": caption, "parse_mode": "Markdown"}
                )
            else:
                await self.http_client.post(
                    f"{base_url}/sendMessage",
                    json={"chat_id": settings.telegram_chat_id, "text": caption, "parse_mode": "Markdown"}
                )
            logger.success(f"✅ Alerta enviado: {product.product_name}")
        except Exception as e:
            logger.error(f"Erro no Telegram: {e}")

    async def run(self):
        logger.info("🕵️ Agente Iniciando Varredura...")
        try:
            data = await self.search_trends("produtos virais tiktok shopee brasil gadgets tech")
        except Exception as e:
            logger.error(f"Falha na busca Serper: {e}")
            return

        analysis = await self.analyze_market(data)
        logger.info(f"📊 Encontradas {len(analysis.top_opportunities)} oportunidades.")

        for item in analysis.top_opportunities:
            # 1. Verifica Memória
            if await self.check_memory(item.product_name):
                continue
            
            # 2. Filtra Score Baixo
            if item.virality_score < 75:
                continue

            # 3. Gera Imagem
            img_url = await self.generate_viral_image(item.visual_prompt)
            
            # 4. Envia e Salva
            await self.send_alert(item, img_url)
            await self.save_to_memory(item)
            
            await asyncio.sleep(5)

    async def start_loop(self):
        await init_db()
        logger.info("🚀 SISTEMA OPERACIONAL - MONITORANDO 24/7")
        while True:
            await self.run()
            logger.info("💤 Standby por 6 horas...")
            await asyncio.sleep(6 * 3600)

if __name__ == "__main__":
    try:
        agent = MarketingAgent()
        asyncio.run(agent.start_loop())
    except KeyboardInterrupt:
        logger.warning("Sistema desligado manualmente.")