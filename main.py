import asyncio
import json
import os
import random
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

# --- MOVIEPY: O Editor de Vídeo ---
try:
    # Tenta importar. Se falhar (localmente sem ffmpeg), o bot não quebra.
    from moviepy.editor import ImageClip
    HAS_VIDEO = True
except ImportError:
    HAS_VIDEO = False
    logger.warning("⚠️ MoviePy não encontrado. O agente enviará apenas imagens.")

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

# --- 3. MODELOS DE DADOS ---
class ProductOpportunity(BaseModel):
    product_name: str = Field(..., description="Nome curto do produto.")
    virality_score: int = Field(..., ge=0, le=100)
    target_audience: str
    marketing_hook: str
    visual_prompt: str = Field(..., description="Prompt visual em INGLÊS. Detalhado, cinematic, 8k.")
    hashtags: List[str]
    source_url: Optional[str] = Field(None, description="URL do produto se encontrada.")

class MarketAnalysisResult(BaseModel):
    top_opportunities: List[ProductOpportunity]
    market_mood: Literal["Alta Demanda", "Saturado", "Emergente", "Estável"]
    strategy_advice: str

# --- 4. LISTA DE NICHOS (Rotação) ---
NICHE_QUERIES = [
    "gadgets cozinha viral tiktok",
    "acessórios setup gamer rgb barato",
    "organizadores casa inteligentes",
    "produtos limpeza satisfatoria",
    "brinquedos pets interativos",
    "ferramentas multifuncionais diy",
    "acessórios carro viagem conforto"
]

# --- 5. LÓGICA DE VÍDEO (Video Factory) ---
def create_video_from_image(image_path: str, output_path: str):
    """
    Transforma uma imagem estática em um vídeo MP4 de 5 segundos.
    Roda em Thread separada para não travar o bot.
    """
    if not HAS_VIDEO: return False
    try:
        # Cria um clipe de 5 segundos
        clip = ImageClip(image_path).set_duration(5)
        clip = clip.set_fps(24)
        
        # Renderiza (Preset ultrafast para economizar CPU do Railway)
        clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio=False, 
            preset="ultrafast",
            threads=4,
            logger=None # Silencia logs do ffmpeg
        )
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao renderizar vídeo: {e}")
        return False

# --- 6. O AGENTE ---
class MarketingAgent:
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=40.0, follow_redirects=True)
        self.reasoning_client = instructor.from_openai(
            AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            ),
            mode=instructor.Mode.JSON
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_trends(self, query: str) -> dict:
        logger.info(f"🔍 [COLETA] Google Trends: '{query}'")
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "gl": "br", "hl": "pt-br", "num": 8, "tbs": "qdr:m"})
        headers = {'X-API-KEY': settings.serper_api_key, 'Content-Type': 'application/json'}
        response = await self.http_client.post(url, headers=headers, content=payload)
        response.raise_for_status()
        return response.json()

    async def analyze_market(self, search_data: dict) -> MarketAnalysisResult:
        logger.info("🧠 [ANÁLISE] Processando via LLM...")
        organic = search_data.get("organic", [])
        if not organic: return MarketAnalysisResult(top_opportunities=[], market_mood="Estável", strategy_advice="N/A")
        
        snippets = "\n".join([f"- {item.get('title')} (Link: {item.get('link')}): {item.get('snippet')}" for item in organic])
        
        return await self.reasoning_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            response_model=MarketAnalysisResult,
            messages=[
                {"role": "system", "content": "Você é um Caçador de Produtos Virais. Identifique produtos físicos."},
                {"role": "user", "content": f"Analise estes resultados. Extraia o produto e a URL de venda:\n{snippets}"}
            ],
        )

    async def process_media(self, prompt: str) -> tuple[Optional[str], Optional[str]]:
        """Gera imagem e converte em vídeo."""
        logger.info(f"🎨 [ASSET] Gerando mídia para: {prompt[:15]}...")
        img_filename = f"temp_{random.randint(1000,9999)}.jpg"
        vid_filename = f"video_{random.randint(1000,9999)}.mp4"
        
        try:
            # 1. Gera URL da Imagem (Pollinations)
            import urllib.parse
            encoded = urllib.parse.quote(f"hyper realistic product photography, 8k, cinematic lighting, {prompt}")
            image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1350&nologo=true"
            
            # 2. Baixa a Imagem
            resp = await self.http_client.get(image_url)
            if resp.status_code == 200:
                with open(img_filename, "wb") as f:
                    f.write(resp.content)
                
                # 3. Converte para Vídeo (em thread separada)
                loop = asyncio.get_running_loop()
                success = await loop.run_in_executor(None, create_video_from_image, img_filename, vid_filename)
                
                if success:
                    return image_url, vid_filename
                
                # Se falhar o vídeo, retorna só a imagem local (ou limpa ela)
                if os.path.exists(img_filename): os.remove(img_filename)
            
            return image_url, None
        except Exception as e:
            logger.error(f"⚠️ Erro de mídia: {e}")
            return None, None

    async def send_alert(self, product: ProductOpportunity, image_url: str, video_path: str):
        base_url = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
        link = f"\n🔗 [Ver Oferta]({product.source_url})" if product.source_url else ""
        caption = (
            f"🎬 **{product.product_name.upper()}**\n"
            f"🔥 Score: {product.virality_score}/100\n\n"
            f"🎯 {product.marketing_hook}\n"
            f"🏷️ `{' '.join(product.hashtags[:5])}`"
            f"{link}"
        )
        
        try:
            if video_path and os.path.exists(video_path):
                logger.info("📤 Enviando VÍDEO para o Telegram...")
                with open(video_path, "rb") as v:
                    await self.http_client.post(
                        f"{base_url}/sendVideo",
                        data={"chat_id": settings.telegram_chat_id, "caption": caption, "parse_mode": "Markdown"},
                        files={"video": v},
                        timeout=120.0 # Upload de vídeo demora mais
                    )
                # Limpa o arquivo de vídeo após envio
                os.remove(video_path)
            elif image_url:
                logger.info("📤 Enviando IMAGEM (Fallback)...")
                await self.http_client.post(
                    f"{base_url}/sendPhoto",
                    json={"chat_id": settings.telegram_chat_id, "photo": image_url, "caption": caption, "parse_mode": "Markdown"}
                )
            
            # Limpa imagem temporária se sobrou
            for f in os.listdir():
                if f.startswith("temp_") and f.endswith(".jpg"):
                    try: os.remove(f)
                    except: pass
                    
            logger.success(f"✅ Enviado: {product.product_name}")
        except Exception as e:
            logger.error(f"Erro Telegram: {e}")

    async def check_memory(self, product_name: str) -> bool:
        async with AsyncSession(engine) as session:
            result = await session.execute(select(SentProduct).where(SentProduct.product_name == product_name))
            if result.scalars().first():
                logger.info(f"🚫 [SKIP] Já enviado: {product_name}")
                return True
            return False

    async def save_memory(self, product: ProductOpportunity):
        async with AsyncSession(engine) as session:
            session.add(SentProduct(product_name=product.product_name, virality_score=product.virality_score))
            await session.commit()

    async def run(self):
        topic = random.choice(NICHE_QUERIES)
        logger.info(f"🎲 Sorteio: {topic}")
        try:
            data = await self.search_trends(topic)
            analysis = await self.analyze_market(data)
            
            for item in analysis.top_opportunities:
                if await self.check_memory(item.product_name): continue
                if item.virality_score < 75: continue
                
                # Gera Imagem e Tenta Vídeo
                img_url, vid_path = await self.process_media(item.visual_prompt)
                
                await self.send_alert(item, img_url, vid_path)
                await self.save_memory(item)
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Falha no ciclo: {e}")

    async def start_loop(self):
        await init_db()
        logger.info("🚀 AGENTE DE VÍDEO (RAILWAY) - ATIVO")
        while True:
            await self.run()
            logger.info("💤 Dormindo 6h...")
            await asyncio.sleep(6 * 3600)

if __name__ == "__main__":
    try:
        asyncio.run(MarketingAgent().start_loop())
    except KeyboardInterrupt:
        pass