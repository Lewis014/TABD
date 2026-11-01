"""
Vectorización común a Colab y a los scripts locales.

Cada modelo se carga y se usa de una sola forma, para que corpus, consultas y
títulos se vectoricen igual en cualquier fase. Los roles son:
  doc          abstract del corpus
  query        abstract usado como consulta
  short_query  título usado como consulta

SciNCL y SPECTER2 esperan normalmente "título [SEP] abstract". Aquí reciben solo el
abstract, igual que los demás modelos: indexar el título filtraría la respuesta en la
evaluación título->abstract.
"""

import numpy as np
import torch

import config as cfg


class Specter2:
    """SPECTER2 con un adaptador por rol: vector del primer token de la última capa."""

    def __init__(self, spec, device):
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        self.device = device
        self.roles = {role: name.split("/")[-1] for role, name in spec["adapters"].items()}
        self.tokenizer = AutoTokenizer.from_pretrained(spec["hf"])
        self.model = AutoAdapterModel.from_pretrained(spec["hf"])
        for name in set(spec["adapters"].values()):
            self.model.load_adapter(name, source="hf", load_as=name.split("/")[-1], set_active=True)
        self.model.to(device).eval()

    @torch.no_grad()
    def encode(self, texts, role, show_progress=True):
        self.model.set_active_adapters(self.roles[role])
        out = []
        for i in range(0, len(texts), cfg.BATCH_SIZE):
            batch = self.tokenizer(
                texts[i:i + cfg.BATCH_SIZE], padding=True, truncation=True,
                max_length=cfg.MAX_SEQ_LEN, return_token_type_ids=False,
                return_tensors="pt",
            ).to(self.device)
            out.append(self.model(**batch).last_hidden_state[:, 0, :].cpu().numpy())
            if show_progress and i % (cfg.BATCH_SIZE * 200) == 0:
                print(f"  {i:,} / {len(texts):,}", flush=True)
        return np.concatenate(out).astype(np.float32)


def load_encoder(model_key, device):
    spec = cfg.ALL_MODELS[model_key]
    if "adapters" in spec:
        return Specter2(spec, device)

    # Aquí y no arriba: SPECTER2 corre en un entorno que no siempre la tiene compatible
    from sentence_transformers import SentenceTransformer, models

    # Los BERT puros se envuelven con un pooling explícito para obtener un único
    # vector por texto. E5 ya trae su propio pooling y normalización.
    if "pooling" in spec:
        word = models.Transformer(spec["hf"], max_seq_length=cfg.MAX_SEQ_LEN)
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode=spec["pooling"])
        return SentenceTransformer(modules=[word, pool], device=device)
    model = SentenceTransformer(spec["hf"], device=device)
    model.max_seq_length = cfg.MAX_SEQ_LEN
    return model


def encode(model_key, model, texts, role, device, show_progress=True) -> np.ndarray:
    spec = cfg.ALL_MODELS[model_key]
    if "adapters" in spec:
        return model.encode(texts, role, show_progress)
    prefix = spec.get("prefixes", {}).get(role, "")
    emb = model.encode(
        [prefix + t for t in texts],
        batch_size=cfg.BATCH_SIZE,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,   # la normalización para coseno se hace en FAISS
        device=device,
    )
    return emb.astype(np.float32)
