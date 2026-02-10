import asyncio
import json
import os
import random
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional

import httpx
import instructor
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlmodel import Field as DBField, SQLModel, select, desc
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from tenacity import retry, stop_after_attempt, wait_exponential

# --- SERVIDOR FALSO PARA O RENDER ---
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Agente Rodando 100%")

def start_fake_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    logger.info(f"🌍 Servidor Fake rodando na porta {port}")
    server.serve_forever()

# --- 1. CONFIGURAÇÃO ---

class Settings(BaseSettings):
    serper_api_key: str
    openrouter_api_key: str  
    telegram_bot_token: str
    telegram_chat_id: str
    database_url: str = "sqlite+aiosqlite:///marketing_agent.db"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
engine = create_async_engine(settings.database_url, echo=False)

# --- 2. BANCO DE DADOS ---

class SentProduct(SQLModel, table=True):
    id: Optional[int] = DBField(default=None, primary_key=True)
    product_name: str = DBField(index=True)
    virality_score: int
    sent_at: datetime = DBField(default_factory=lambda: datetime.now(timezone.utc))

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

# --- 3. ESTRUTURA DE DADOS OTIMIZADA ---

class ProductOpportunity(BaseModel):
    product_name: str = Field(..., description="Nome comercial curto do produto.")
    virality_score: int = Field(..., ge=0, le=100)
    target_audience: str
    pain_point_solved: str
    
    # SUGESTÃO B: Prompt Engineering embutido na tipagem
    visual_prompt: str = Field(..., description="Prompt visual em INGLÊS. OBRIGATÓRIO incluir: 'Professional product photography, studio lighting, bokeh background, 8k resolution, cinematic shot'. Descreva o produto em uso.")
    
    marketing_hook: str
    hashtags: List[str]
    
    # SUGESTÃO A: Captura de URL para validação
    source_url: Optional[str] = Field(None, description="A URL do produto encontrada na busca (Shopee, Amazon, etc).")

class MarketAnalysisResult(BaseModel):
    top_opportunities: List[ProductOpportunity]
    market_mood: Literal["Alta Demanda", "Saturado", "Emergente", "Estável"]
    strategy_advice: str

# --- 4. LISTA DE NICHOS ---
NICHE_QUERIES = [
    "acessórios setup gamer barato led rgb",
    "gadgets produtividade home office shopee",
    "suporte celular articulado mesa review",
    "utensílios cozinha silicone viral tiktok",
    "organizadores geladeira acrilico",
    "mini processador alimentos eletrico portatil",
    "mop giratorio limpeza pratica review",
    "aspirador po portatil carro potente",
    "massageador pescoço elétrico relaxamento",
    "brinquedos interativos gatos laser"
]

# --- 5. O AGENTE ---

class MarketingAgent:
    def __init__(self):
        # Timeouts maiores para evitar falsos negativos na validação de link
        self.http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self.reasoning_client = instructor.from_openai(
            AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            ),
            mode=instructor.Mode.JSON
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_trends(self, query: str) -> dict:
        logger.info(f"🔍 [COLETA] Sondando Google Trends: '{query}'")
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "gl": "br", "hl": "pt-br", "num": 10, "tbs": "qdr:m"})
        headers = {'X-API-KEY': settings.serper_api_key, 'Content-Type': 'application/json'}
        
        response = await self.http_client.post(url, headers=headers, content=payload)
        response.raise_for_status()
        return response.json()

    # SUGESTÃO A: Validação de URL (Sanity Check)
    async def verify_url_integrity(self, url: str) -> bool:
        """Verifica se o link realmente existe (Status 200)."""
        if not url:
            return True # Se não tem link, aprovamos a ideia (o usuário busca depois)
        
        logger.info(f"🛡️ Validando link: {url[:30]}...")
        try:
            # User-Agent é crucial para não ser bloqueado por Amazon/Shopee
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = await self.http_client.head(url, headers=headers)
            
            if response.status_code < 400:
                return True
            
            # Alguns sites não aceitam HEAD, tentamos GET rápido
            response = await self.http_client.get(url, headers=headers)
            return response.status_code < 400
            
        except Exception as e:
            logger.warning(f"⚠️ Link suspeito ou inacessível: {e}")
            return False # Link quebrado? Descarta o produto.

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def analyze_market(self, search_data: dict) -> MarketAnalysisResult:
        logger.info("🧠 [ANÁLISE] Raciocinando sobre os dados...")
        organic = search_data.get("organic", [])
        if not organic:
            return MarketAnalysisResult(top_opportunities=[], market_mood="Estável", strategy_advice="Sem dados.")

        # ATUALIZAÇÃO: Injetamos o 'link' no texto para o LLM poder extrair
        snippets = "\n".join([f"- {item.get('title')} (Link: {item.get('link')}): {item.get('snippet')}" for item in organic])
        
        return await self.reasoning_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            response_model=MarketAnalysisResult,
            messages=[
                {"role": "system", "content": "Você é um Caçador de Produtos Virais. Identifique produtos físicos."},
                {"role": "user", "content": f"Analise estes resultados. Extraia o produto e a URL de venda se houver:\n{snippets}"}
            ],
        )

    async def generate_viral_image(self, prompt: str) -> Optional[str]:
        logger.info(f"🎨 [ARTE] Gerando imagem: {prompt[:30]}...")
        try:
            import urllib.parse
            # SUGESTÃO B: Reforço no prompt via código também
            enhanced_prompt = f"hyper realistic, 8k, bokeh, {prompt}"
            encoded_prompt = urllib.parse.quote(enhanced_prompt)
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
            return image_url
        except Exception as e:
            logger.error(f"⚠️ Falha na imagem: {e}")
            return None

    async def check_memory(self, product_name: str) -> bool:
        async with AsyncSession(engine) as session:
            statement = select(SentProduct).where(
                SentProduct.product_name == product_name,
                SentProduct.sent_at > datetime.now(timezone.utc) - timedelta(days=7)
            )
            result = await session.execute(statement)
            if result.scalars().first():
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
        
        # Link clicável se existir
        link_text = f"\n🔗 [Ver Produto]({product.source_url})" if product.source_url else ""
        
        caption = (
            f"🔥 **{product.product_name.upper()}** (Score: {product.virality_score})\n\n"
            f"🎯 **Público:** {product.target_audience}\n"
            f"💡 **Hook:** {product.marketing_hook}\n"
            f"🏷️ `{' '.join(product.hashtags[:5])}`"
            f"{link_text}"
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

    async def send_weekly_report(self):
        logger.info("📅 Gerando Relatório Semanal...")
        async with AsyncSession(engine) as session:
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            statement = (
                select(SentProduct)
                .where(SentProduct.sent_at >= seven_days_ago)
                .order_by(desc(SentProduct.virality_score))
                .limit(5)
            )
            result = await session.execute(statement)
            top_products = result.scalars().all()

            if not top_products:
                return

            msg = "🏆 **TOP 5 DA SEMANA** 🏆\n\n"
            for i, p in enumerate(top_products, 1):
                msg += f"{i}️⃣ **{p.product_name}** ({p.virality_score}/100)\n"
            
            await self.http_client.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                json={"chat_id": settings.telegram_chat_id, "text": msg, "parse_mode": "Markdown"}
            )

    async def run(self):
        current_query = random.choice(NICHE_QUERIES)
        logger.info(f"🎲 Sorteio do Ciclo: '{current_query}'")
        
        try:
            data = await self.search_trends(current_query)
        except Exception as e:
            logger.error(f"Falha na busca Serper: {e}")
            return

        analysis = await self.analyze_market(data)
        logger.info(f"📊 Encontradas {len(analysis.top_opportunities)} oportunidades.")

        for item in analysis.top_opportunities:
            # 1. Memória
            if await self.check_memory(item.product_name):
                continue
            
            # 2. Score
            if item.virality_score < 75:
                continue

            # 3. NOVO: Validação de Link
            if item.source_url:
                is_valid = await self.verify_url_integrity(item.source_url)
                if not is_valid:
                    logger.warning(f"❌ Link quebrado detectado para {item.product_name}. Pulando...")
                    continue

            # 4. Ação
            img_url = await self.generate_viral_image(item.visual_prompt)
            await self.send_alert(item, img_url)
            await self.save_to_memory(item)
            await asyncio.sleep(5)

    async def start_loop(self):
        await init_db()
        logger.info("🚀 SISTEMA OPERACIONAL - MONITORANDO 24/7")
        # await self.send_weekly_report() # Descomente se quiser relatório ao iniciar
        
        while True:
            await self.run()
            logger.info("💤 Standby por 6 horas...")
            await asyncio.sleep(6 * 3600)

if __name__ == "__main__":
    threading.Thread(target=start_fake_server, daemon=True).start()
    try:
        agent = MarketingAgent()
        asyncio.run(agent.start_loop())
    except KeyboardInterrupt:
        logger.warning("Sistema desligado manualmente.")