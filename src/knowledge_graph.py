# src/knowledge_graph.py
"""
Knowledge Graph — Extracts entities from paper chunks using regex patterns.
Builds a multi-paper entity graph using NetworkX and renders with Pyvis.
"""

import re
import os
import json
import networkx as nx
from pyvis.network import Network
from typing import List, Dict, Set

from config import KG_OUTPUT_PATH

# ── Entity extraction patterns ─────────────────────────────────────────────────

MODEL_PATTERNS = [
    r"\b(BERT|RoBERTa|ALBERT|DistilBERT|XLNet|ELECTRA|DeBERTa|"
    r"GPT[-\s]?[234]?|GPT-4|ChatGPT|InstructGPT|"
    r"T5|mT5|BART|PEGASUS|ProphetNet|"
    r"ViT|DeiT|Swin|BEiT|MAE|DINO|CLIP|ALIGN|"
    r"ResNet[-\s]?\d*|VGG[-\s]?\d*|EfficientNet[-\s]?\w*|DenseNet[-\s]?\d*|"
    r"MobileNet[-\s]?\w*|InceptionV\d|AlexNet|"
    r"LSTM|GRU|BiLSTM|Seq2Seq|Transformer|"
    r"Llama[-\s]?\d*|LLaMA[-\s]?\d*|Mistral|Gemma[-\s]?\d*|Phi[-\s]?\d*|"
    r"PaLM|Gemini|Claude|GPT-4o|"
    r"DALL[-·]E|Stable Diffusion|Imagen|"
    r"Whisper|wav2vec|HuBERT|"
    r"SVM|Random Forest|XGBoost|LightGBM|CatBoost|AdaBoost|"
    r"YOLO[-\s]?v?\d*|Faster[-\s]?RCNN|RetinaNet|Mask[-\s]?RCNN|"
    r"U-Net|DeepLab[-\s]?\w*|SegNet)\b"
]

DATASET_PATTERNS = [
    r"\b(ImageNet|COCO|MS[-\s]?COCO|"
    r"CIFAR[-\s]?10|CIFAR[-\s]?100|MNIST|Fashion[-\s]?MNIST|SVHN|"
    r"SQuAD|SQuAD\s?2|GLUE|SuperGLUE|MultiNLI|SNLI|"
    r"WikiText[-\s]?\d*|Penn Treebank|PTB|"
    r"WMT\d*|CommonCrawl|OpenWebText|BookCorpus|C4|LAION|"
    r"MS[-\s]?MARCO|TriviaQA|Natural Questions|HotpotQA|"
    r"VOC\s?\d*|ADE20K|Cityscapes|"
    r"LibriSpeech|VoxCeleb\d*|"
    r"MIMIC[-\s]?III|ChestX-ray\d*|CheXpert)\b"
]

METRIC_PATTERNS = [
    r"\b(accuracy|F1[-\s]?score|F1|precision|recall|"
    r"BLEU|ROUGE[-\s]?\w*|METEOR|CIDEr|SPICE|"
    r"perplexity|PPL|"
    r"mAP|AP\d*|IoU|"
    r"AUC|ROC|AUC[-\s]?ROC|"
    r"RMSE|MAE|MSE|"
    r"WER|CER|"
    r"FID|IS|LPIPS|SSIM|PSNR|"
    r"top[-\s]?1|top[-\s]?5)\b"
]


def extract_entities(text: str) -> Dict[str, Set[str]]:
    """Extract model, dataset, and metric mentions from text."""
    entities: Dict[str, Set[str]] = {
        "model":   set(),
        "dataset": set(),
        "metric":  set(),
    }
    for pattern in MODEL_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities["model"].add(m.group(0).strip())
    for pattern in DATASET_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities["dataset"].add(m.group(0).strip())
    for pattern in METRIC_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities["metric"].add(m.group(0).lower().strip())
    return entities


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    # ── Build ──────────────────────────────────────────────────────────────────

    def add_paper(self, chunks: List[Dict]) -> None:
        """Processes chunks from one paper and adds all entities + edges."""
        if not chunks:
            return
        source = chunks[0]["source"]

        if not self.graph.has_node(source):
            self.graph.add_node(
                source,
                label=source[:30] + "…" if len(source) > 30 else source,
                type="paper",
                color="#6c5ce7",
                size=35,
            )

        all_entities: Dict[str, Set[str]] = {
            "model":   set(),
            "dataset": set(),
            "metric":  set(),
        }
        for chunk in chunks:
            found = extract_entities(chunk["text"])
            for etype, names in found.items():
                all_entities[etype].update(names)

        color_map = {
            "model":   "#00b894",
            "dataset": "#fdcb6e",
            "metric":  "#e17055",
        }

        for etype, names in all_entities.items():
            for name in names:
                node_id = f"{etype}::{name}"
                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id,
                        label=name,
                        type=etype,
                        color=color_map[etype],
                        size=20,
                    )
                if not self.graph.has_edge(source, node_id):
                    self.graph.add_edge(source, node_id, weight=1, relation=etype)
                else:
                    self.graph[source][node_id]["weight"] += 1

    # ── Queries ────────────────────────────────────────────────────────────────

    def query_entity(self, entity_name: str) -> List[str]:
        """Return all paper nodes that mention a given entity (partial match)."""
        papers = []
        for node_id in self.graph.nodes:
            label = self.graph.nodes[node_id].get("label", "")
            if entity_name.lower() in label.lower():
                for neighbor in self.graph.neighbors(node_id):
                    if self.graph.nodes[neighbor].get("type") == "paper":
                        papers.append(neighbor)
        return list(set(papers))

    def compare_models(self, model_a: str, model_b: str) -> Dict:
        """Find papers that mention both models."""
        papers_a = set(self.query_entity(model_a))
        papers_b = set(self.query_entity(model_b))
        both     = papers_a & papers_b
        return {
            "model_a":          model_a,
            "model_b":          model_b,
            "papers_with_both": list(both),
            "only_a":           list(papers_a - both),
            "only_b":           list(papers_b - both),
        }

    def get_summary(self) -> Dict:
        """Returns graph statistics and entity lists."""
        buckets: Dict[str, List] = {
            "paper":   [],
            "model":   [],
            "dataset": [],
            "metric":  [],
        }
        for node, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            if t in buckets:
                buckets[t].append(node)

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "papers":      buckets["paper"],
            "models":      [self.graph.nodes[n].get("label", n) for n in buckets["model"]],
            "datasets":    [self.graph.nodes[n].get("label", n) for n in buckets["dataset"]],
            "metrics":     [self.graph.nodes[n].get("label", n) for n in buckets["metric"]],
        }

    # ── Render ─────────────────────────────────────────────────────────────────

    def render_html(self, output_path: str = KG_OUTPUT_PATH) -> str:
        """
        Renders the graph as a rich interactive Pyvis HTML file.
        Fixed: proper physics, large nodes, centered layout, legend.
        """
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        net = Network(
            height="620px",
            width="100%",
            bgcolor="#0f0f1a",
            font_color="#ffffff",
            notebook=False,
            directed=False,
        )

        # ── Physics: spread nodes out properly ────────────────────────────────
        physics_opts = {
            "physics": {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -120,
                    "centralGravity": 0.005,
                    "springLength": 220,
                    "springConstant": 0.08,
                    "damping": 0.6,
                    "avoidOverlap": 1.0
                },
                "stabilization": {
                    "enabled": True,
                    "iterations": 300,
                    "updateInterval": 25
                },
                "minVelocity": 0.75
            }
        }
        net.set_options(json.dumps(physics_opts))

        # ── Add nodes ─────────────────────────────────────────────────────────
        for node_id, data in self.graph.nodes(data=True):
            ntype = data.get("type", "unknown")
            label = data.get("label", node_id)
            color = data.get("color", "#888888")
            size  = data.get("size", 20)

            type_icons = {
                "paper":   "📄",
                "model":   "🤖",
                "dataset": "🗃️",
                "metric":  "📊",
            }
            icon = type_icons.get(ntype, "•")

            tooltip = (
                f"<div style='font-family:sans-serif;padding:6px'>"
                f"<b>{icon} {label}</b><br>"
                f"<span style='color:#aaa'>Type: {ntype}</span>"
                f"</div>"
            )

            net.add_node(
                node_id,
                label=label,
                color={
                    "background": color,
                    "border":     "#ffffff",
                    "highlight":  {"background": "#ffffff", "border": color},
                    "hover":      {"background": "#ffffff", "border": color},
                },
                size=size,
                title=tooltip,
                font={
                    "size":  14 if ntype == "paper" else 12,
                    "color": "#ffffff",
                    "bold":  ntype == "paper",
                },
                borderWidth=2,
                shadow=True,
            )

        # ── Add edges ─────────────────────────────────────────────────────────
        edge_colors = {
            "model":   "#00b894",
            "dataset": "#fdcb6e",
            "metric":  "#e17055",
        }
        for u, v, data in self.graph.edges(data=True):
            rel   = data.get("relation", "")
            width = min(data.get("weight", 1) * 1.5, 6)
            net.add_edge(
                u, v,
                title=rel,
                width=width,
                color={"color": edge_colors.get(rel, "#666699"), "opacity": 0.7},
                smooth={"type": "dynamic"},
            )

        # ── Write and inject legend + title ───────────────────────────────────
        net.write_html(output_path)

        # Inject legend HTML into the rendered file
        legend_html = """
<div id="kg-legend" style="
    position:absolute; top:14px; left:14px;
    background:rgba(15,15,26,0.92);
    border:1px solid #444;
    border-radius:10px;
    padding:12px 16px;
    font-family:sans-serif;
    font-size:13px;
    color:#fff;
    z-index:999;
    pointer-events:none;
">
  <div style="font-weight:bold;margin-bottom:8px;font-size:14px;">📌 Legend</div>
  <div style="display:flex;align-items:center;gap:8px;margin:4px 0">
    <span style="width:14px;height:14px;border-radius:50%;background:#6c5ce7;display:inline-block"></span> Paper
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin:4px 0">
    <span style="width:14px;height:14px;border-radius:50%;background:#00b894;display:inline-block"></span> Model
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin:4px 0">
    <span style="width:14px;height:14px;border-radius:50%;background:#fdcb6e;display:inline-block"></span> Dataset
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin:4px 0">
    <span style="width:14px;height:14px;border-radius:50%;background:#e17055;display:inline-block"></span> Metric
  </div>
  <div style="margin-top:8px;font-size:11px;color:#aaa">Drag nodes • Scroll to zoom</div>
</div>
"""
        # Inject legend before closing </body>
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()

        html = html.replace(
            "<div id=\"mynetwork\"",
            f"{legend_html}<div id=\"mynetwork\" style=\"position:relative\""
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    # ── Reset ──────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Resets the graph entirely."""
        self.graph = nx.Graph()
