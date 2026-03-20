"""
DFU Classification + Latent Space Analysis — Pipeline Completo
===============================================================
Tarefa      : Classificação binária de feridas de pé diabético (isquemia)
Modelos     : EfficientNet-B0 ou ResNet-50 (fine-tuning ImageNet)
Novidades   : Extração de features (penúltima camada) → PCA → Principal Curves
              → Métricas geométricas → Relatório consolidado

Estrutura de diretórios esperada:
    ../data/ischaemia/
        Aug-Positive/
        Aug-Negative/
    ../reports/
        metrics/
        figures/
        figures/pca/
        emissions/
    ../models/ischaemia/

Dependências adicionais ao ambiente original:
    pip install principal-curves codecarbon

Uso:
    python dfu_pipeline_complete.py
"""

# ═══════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Ellipse
from matplotlib.ticker import MaxNLocator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from PIL import Image

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score,
    confusion_matrix, silhouette_score
)

from scipy.stats import mannwhitneyu

import codecarbon
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ═══════════════════════════════════════════════════════════════
# Configurações globais
# ═══════════════════════════════════════════════════════════════
TASK             = "Ischaemia"    # "Infection" ou "Ischaemia"
IMG_SIZE         = 256
BATCH_SIZE       = 32
EPOCHS           = 100
MODEL_NAME       = "efficientnet" # "efficientnet" ou "resnet50"
PCA_N_COMPONENTS = 50             # refinado via scree plot
PC_CURVE_K       = 5              # nós de suavização da curva principal
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIR_MODELS    = "../models/ischaemia"
DIR_METRICS   = "../reports/metrics"
DIR_FIGURES   = "../reports/figures"
DIR_PCA_FIG   = "../reports/figures/pca"
DIR_EMISSIONS = "../reports/emissions"

for d in [DIR_MODELS, DIR_METRICS, DIR_FIGURES, DIR_PCA_FIG, DIR_EMISSIONS]:
    os.makedirs(d, exist_ok=True)

# Paleta consistente em todo o pipeline
FOLD_COLORS  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
CLASS_COLORS = {'pos': '#E8593C', 'neg': '#3B8BD4'}
METRICS_DISPLAY = {
    'auc'      : 'AUC-ROC',
    'pr_auc'   : 'PR-AUC',
    'f1'       : 'F1-Score',
    'accuracy' : 'Accuracy',
    'precision': 'Precision',
    'recall'   : 'Recall',
}

# ═══════════════════════════════════════════════════════════════
# 1. Pré-processamento de imagens
# ═══════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ═══════════════════════════════════════════════════════════════
# 2. Dataset customizado
# ═══════════════════════════════════════════════════════════════
class DFUDataset(Dataset):
    """
    Dataset de feridas de pé diabético.

    Estrutura esperada:
        root_dir/
            Aug-Positive/   ← imagens com rótulo 1
            Aug-Negative/   ← imagens com rótulo 0

    Identificadores de paciente são extraídos do prefixo do nome de arquivo
    (ex: "pat042_1X_M.jpg" → "pat042") e usados no StratifiedGroupKFold para
    garantir que o mesmo paciente não apareça em treino e validação ao mesmo tempo.
    """
    def __init__(self, root_dir, task=TASK, transform=None):
        if not isinstance(root_dir, str):
            raise TypeError(f"root_dir deve ser str, recebido: {type(root_dir).__name__}")

        self.root_dir    = root_dir
        self.task        = task
        self.transform   = transform
        self.image_paths = []
        self.labels      = []
        self.identifiers = []

        print(f"[DFUDataset] TASK: {task}")

        positive_dir = os.path.join(root_dir, "Aug-Positive")
        negative_dir = os.path.join(root_dir, "Aug-Negative")

        for img_name in os.listdir(positive_dir):
            self.image_paths.append(os.path.join(positive_dir, img_name))
            self.labels.append(1)
            self.identifiers.append(img_name.split('_')[0])

        for img_name in os.listdir(negative_dir):
            self.image_paths.append(os.path.join(negative_dir, img_name))
            self.labels.append(0)
            self.identifiers.append(img_name.split('_')[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ═══════════════════════════════════════════════════════════════
# 3. Criação do modelo
# ═══════════════════════════════════════════════════════════════
def create_model():
    """
    EfficientNet-B0 : Dropout(0.4) → Linear(1280 → 1)
    ResNet-50       : Dropout(0.5) → Linear(2048 → 1)

    A penúltima camada (avgpool) produz features de dimensão
    1280 (EfficientNet) ou 2048 (ResNet-50) — capturadas pelo hook.
    """
    if MODEL_NAME == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))

    elif MODEL_NAME == "efficientnet":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, 1))

    return model.to(DEVICE)

# ═══════════════════════════════════════════════════════════════
# 4. Métricas de classificação
# ═══════════════════════════════════════════════════════════════
def calculate_metrics(true, preds):
    preds_class = (preds > 0.5).astype(int)
    return {
        'auc'             : roc_auc_score(true, preds),
        'pr_auc'          : average_precision_score(true, preds),
        'f1'              : f1_score(true, preds_class),
        'accuracy'        : accuracy_score(true, preds_class),
        'precision'       : precision_score(true, preds_class, zero_division=0),
        'recall'          : recall_score(true, preds_class, zero_division=0),
        'confusion_matrix': confusion_matrix(true, preds_class)
    }

# ═══════════════════════════════════════════════════════════════
# 5. Avaliação do modelo
#    @track_emissions REMOVIDO daqui — emissões medidas por fold
#    no cross_validation via EmissionsTracker manual
# ═══════════════════════════════════════════════════════════════
def evaluate_model(model, loader, criterion):
    """
    Avalia o modelo num DataLoader.
    Retorna (loss_médio, dicionário de métricas).
    """
    model.eval()
    losses, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            outputs = model(inputs)
            losses.append(criterion(outputs, labels).item())
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return np.mean(losses), metrics

# ═══════════════════════════════════════════════════════════════
# 6. Treinamento com early stopping
# ═══════════════════════════════════════════════════════════════
def train_model(model, train_loader, val_loader, criterion, optimizer, fold):
    """
    Treina por até EPOCHS épocas com early stopping (patience=3).

    Salva:
        best_model_fold{fold}.pth              ← melhor val_loss (temporário)
        ../models/ischaemia/last_model_fold{fold}.pth ← último epoch

    Retorna: history dict com listas 'train_loss', 'val_loss', 'auc', 'f1'.
    """
    best_loss        = float('inf')
    patience_counter = 0
    patience         = 3
    history          = {'train_loss': [], 'val_loss': [], 'auc': [], 'f1': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        val_loss, val_metrics = evaluate_model(model, val_loader, criterion)
        epoch_loss = running_loss / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['auc'].append(val_metrics['auc'])
        history['f1'].append(val_metrics['f1'])

        print(f'  Fold {fold} | Epoch {epoch+1:3d}/{EPOCHS} | '
              f'Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'AUC: {val_metrics["auc"]:.4f} | F1: {val_metrics["f1"]:.4f}')

        if val_loss < best_loss:
            best_loss        = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  → Early stopping na época {epoch+1}')
                break

    last_path = os.path.join(DIR_MODELS, f"last_model_fold{fold}.pth")
    torch.save(model.state_dict(), last_path)
    return history

# ═══════════════════════════════════════════════════════════════
# 7. Grad-CAM
# ═══════════════════════════════════════════════════════════════
def generate_grad_cam(model, img_tensor, target_layer):
    """
    Gera visualização Grad-CAM para um tensor de imagem.
    target_layer: ex. model.features[-1] para EfficientNet.
    """
    cam = GradCAM(model=model, target_layers=[target_layer],
                  use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0))[0]
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    return show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

# ═══════════════════════════════════════════════════════════════
# 8. Extração de features — forward hook na penúltima camada
# ═══════════════════════════════════════════════════════════════
def extract_features(model, loader):
    """
    Captura a saída de model.avgpool (penúltima camada) via forward hook.

    Como funciona:
        O hook intercepta o tensor de saída do avgpool a cada forward pass
        sem alterar o grafo computacional. Para EfficientNet-B0 a saída é
        (batch, 1280, 1, 1); para ResNet-50, (batch, 2048, 1, 1).
        O flatten produz (batch, D).

    Retorna:
        features_np : ndarray (N, D)
        labels_np   : ndarray (N,)
    """
    features_list = []
    labels_list   = []

    def hook_fn(module, input, output):
        features_list.append(output.flatten(start_dim=1).detach().cpu())

    hook = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for inputs, lbls in loader:
            _ = model(inputs.to(DEVICE))
            labels_list.extend(lbls.numpy())

    hook.remove()  # sempre remover após uso

    return torch.cat(features_list, dim=0).numpy(), np.array(labels_list)

# ═══════════════════════════════════════════════════════════════
# 9. PCA
# ═══════════════════════════════════════════════════════════════
def run_pca(features, n_components=PCA_N_COMPONENTS):
    """
    StandardScaler + PCA.

    Por que normalizar antes?
        Features CNN têm escalas muito diferentes por canal. Sem normalização,
        o PCA é dominado pelas dimensões de maior variância bruta.

    Retorna:
        features_pca : ndarray (N, n_components)
        pca          : objeto sklearn PCA ajustado
        scaler       : objeto StandardScaler ajustado
        explained    : variância explicada por PC
        cumulative   : variância acumulada
    """
    scaler       = StandardScaler()
    features_sc  = scaler.fit_transform(features)
    pca          = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features_sc)
    explained    = pca.explained_variance_ratio_
    cumulative   = np.cumsum(explained)
    return features_pca, pca, scaler, explained, cumulative

# ═══════════════════════════════════════════════════════════════
# 10. Principal Curves
# ═══════════════════════════════════════════════════════════════
def fit_principal_curve(features_2d, labels, fold):
    """
    Ajusta curva principal no espaço PC1-PC2.

    O que é uma curva principal?
        Generalização não-linear do PCA: enquanto a primeira PC é a reta
        que minimiza a distância quadrática média dos pontos, a curva
        principal é a curva suave que minimiza essa mesma distância —
        capturando estrutura não-linear no manifold latente.

    Algoritmo (Hastie & Stuetzle, 1989):
        1. Inicializa com a primeira componente principal.
        2. Projeta cada ponto no ponto mais próximo da curva (pseudotempo λ).
        3. Atualiza cada ponto da curva como E[X | λ(X) = λ].
        4. Repete até convergência.

    Retorna:
        pc          : objeto PrincipalCurve ajustado
        projections : ndarray (N,) — pseudotempo (arclength)
        distances   : ndarray (N,) — distância euclidiana ao manifold
        metrics     : dict com métricas da curva
    """
    try:
        from principal_curves import PrincipalCurve
    except ImportError:
        raise ImportError("Execute: pip install principal-curves")

    pc          = PrincipalCurve(k=PC_CURVE_K)
    pc.fit(features_2d)
    projections = pc.pseudotime_
    distances   = pc.distances_

    metrics = {
        'mean_dist_pos': float(distances[labels == 1].mean()),
        'mean_dist_neg': float(distances[labels == 0].mean()),
        'std_dist_pos' : float(distances[labels == 1].std()),
        'std_dist_neg' : float(distances[labels == 0].std()),
        'curve_length' : float(pc.length_),
        'n_iter'       : int(pc.n_iter_),
    }

    return pc, projections, distances, metrics

# ═══════════════════════════════════════════════════════════════
# 11. Métricas geométricas do espaço latente
# ═══════════════════════════════════════════════════════════════
def compute_geometric_metrics(features_2d, labels, projections, distances):
    """
    Métricas que caracterizam a geometria do espaço latente — todas reportáveis
    em tabela de paper.

    silhouette_score:
        [-1,1]. Compacidade intra-classe vs. separação inter-classe no PCA 2D.
        Valores > 0.3 indicam estrutura clara.

    centroid_distance_pca:
        Distância euclidiana entre centróides das classes no PCA 2D.

    pseudotime_mannwhitney_pval:
        p-valor Mann-Whitney U entre distribuições de pseudotempo das classes.
        p < 0.05 → manifold ordenado que separa as classes.

    bhattacharyya_distance:
        Overlap entre distribuições de pseudotempo.
        Maior = menos overlap = mais separável ao longo da curva.

    mean_dist_to_curve_pos/neg:
        Distância média ao manifold por classe. Positivos mais distantes
        indicam maior heterogeneidade (esperada em isquemia).
    """
    sil = silhouette_score(features_2d, labels) if len(np.unique(labels)) > 1 else float('nan')

    pos_pts = features_2d[labels == 1]
    neg_pts = features_2d[labels == 0]
    centroid_dist = float(np.linalg.norm(pos_pts.mean(axis=0) - neg_pts.mean(axis=0)))

    pseudo_pos = projections[labels == 1]
    pseudo_neg = projections[labels == 0]
    _, pval = mannwhitneyu(pseudo_pos, pseudo_neg, alternative='two-sided')

    bins  = np.linspace(projections.min(), projections.max(), 21)
    h_pos, _ = np.histogram(pseudo_pos, bins=bins, density=True)
    h_neg, _ = np.histogram(pseudo_neg, bins=bins, density=True)
    bhat = float(-np.log(np.sum(np.sqrt(h_pos * h_neg + 1e-12)) * (bins[1] - bins[0])))

    return {
        'silhouette'                  : float(sil),
        'centroid_distance_pca'       : centroid_dist,
        'pseudotime_mannwhitney_pval' : float(pval),
        'bhattacharyya_distance'      : bhat,
        'mean_dist_to_curve_pos'      : float(distances[labels == 1].mean()),
        'mean_dist_to_curve_neg'      : float(distances[labels == 0].mean()),
    }

# ═══════════════════════════════════════════════════════════════
# 12. ReportManager — saídas consolidadas
# ═══════════════════════════════════════════════════════════════
class ReportManager:
    """
    Acumula resultados de cada fold e gera saídas consolidadas ao final.

    Uso:
        reporter = ReportManager()
        for fold ...:
            reporter.log_fold(fold, val_metrics, test_metrics, ...)
        reporter.finalize()

    Saídas:
        CSV único com todas as métricas por fold + linha mean±std
        CSV separado de métricas geométricas
        CSV de emissões por fold
        Painel de classificação (6 métricas × folds)
        Confusion matrices normalizadas (val + test)
        Histórico de treinamento comparativo
        Scree plot comparativo
        PCA scatter comparativo com elipses de confiança
        Pseudotempo: histograma + boxplot por fold
        Emissões: barras por fold + acumulado + equivalências
    """

    def __init__(self):
        self._folds         = []
        self._histories     = []
        self._pca_data      = []
        self._pc_data       = []
        self._conf_matrices = []
        self._emissions     = []
        self._scree_data    = []

    def log_fold(self, fold, val_metrics, test_metrics, geo_metrics,
                 history, features_2d, feat_labels,
                 projections, distances, explained, emissions_kg):
        """Registra os resultados completos de um fold."""
        row = {'fold': fold}
        for k, v in val_metrics.items():
            if k != 'confusion_matrix':
                row[f'val_{k}'] = v
        for k, v in test_metrics.items():
            if k != 'confusion_matrix':
                row[f'test_{k}'] = v
        for k, v in geo_metrics.items():
            row[f'geo_{k}'] = v
        row['emissions_kg_co2'] = emissions_kg

        self._folds.append(row)
        self._histories.append(history)
        self._pca_data.append({'features_2d': features_2d,
                               'labels': feat_labels, 'fold': fold})
        self._pc_data.append({'projections': projections,
                              'distances': distances, 'fold': fold})
        self._conf_matrices.append({'fold': fold,
                                    'val_cm' : val_metrics['confusion_matrix'],
                                    'test_cm': test_metrics['confusion_matrix']})
        self._scree_data.append({'explained': explained, 'fold': fold})
        self._emissions.append({'fold': fold, 'kg_co2': emissions_kg})
        print(f'  [Report] Fold {fold} registrado.')

    def finalize(self):
        """Gera todos os CSVs e figuras. Chamar uma vez ao final do CV."""
        print('\n[Report] Gerando saídas consolidadas...')
        self._export_csv()
        self._plot_classification_panel()
        self._plot_confusion_matrices()
        self._plot_training_histories()
        self._plot_scree_comparative()
        self._plot_pca_comparative()
        self._plot_pseudotime_comparative()
        self._plot_emissions()
        print(f'[Report] Concluído.')
        print(f'  CSVs   → {DIR_METRICS}')
        print(f'  Figuras → {DIR_FIGURES}')
        print(f'  PCA     → {DIR_PCA_FIG}')

    # ── CSV consolidado ────────────────────────────────────────
    def _export_csv(self):
        df = pd.DataFrame(self._folds)

        summary = {'fold': 'mean±std'}
        for col in df.columns:
            if col == 'fold':
                continue
            try:
                summary[col] = f'{df[col].mean():.4f} ± {df[col].std():.4f}'
            except Exception:
                summary[col] = ''

        pd.concat([df, pd.DataFrame([summary])], ignore_index=True).to_csv(
            os.path.join(DIR_METRICS,
                         f'consolidated_{TASK}_{MODEL_NAME}.csv'), index=False
        )

        geo_cols = [c for c in df.columns if c.startswith('geo_')] + ['fold']
        df[geo_cols].to_csv(
            os.path.join(DIR_METRICS,
                         f'geo_metrics_{TASK}_{MODEL_NAME}.csv'), index=False
        )
        print(f'  [CSV] Consolidado e geométrico salvos em {DIR_METRICS}')

    # ── Painel de métricas de classificação ───────────────────
    def _plot_classification_panel(self):
        df    = pd.DataFrame(self._folds)
        folds = df['fold'].astype(int).tolist()

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(
            f'Métricas de classificação por fold — {TASK} ({MODEL_NAME})',
            fontsize=13, y=1.01
        )

        for ax, (key, label) in zip(axes.flat, METRICS_DISPLAY.items()):
            val_vals  = df[f'val_{key}'].tolist()
            test_vals = df[f'test_{key}'].tolist()
            x = np.arange(len(folds))

            bars_v = ax.bar(x - 0.175, val_vals,  0.35, label='Validação',
                            color='#4C72B0', alpha=0.8)
            bars_t = ax.bar(x + 0.175, test_vals, 0.35, label='Teste interno',
                            color='#DD8452', alpha=0.8)

            ax.axhline(np.mean(val_vals),  color='#4C72B0', linestyle='--',
                       linewidth=1.2, alpha=0.6)
            ax.axhline(np.mean(test_vals), color='#DD8452', linestyle='--',
                       linewidth=1.2, alpha=0.6)

            for bar in list(bars_v) + list(bars_t):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7)

            ax.set_title(label, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([f'F{f}' for f in folds], fontsize=8)
            ax.set_ylim(0, 1.12)
            ax.set_ylabel('Score')
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(DIR_FIGURES,
                                 f'classification_panel_{TASK}_{MODEL_NAME}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  [Fig] Painel de classificação salvo.')

    # ── Confusion matrices ─────────────────────────────────────
    def _plot_confusion_matrices(self):
        for split in ['val', 'test']:
            n   = len(self._conf_matrices)
            fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
            fig.suptitle(
                f'Confusion matrices — {split.capitalize()} ({TASK}, {MODEL_NAME})',
                fontsize=12
            )

            for i, entry in enumerate(self._conf_matrices):
                cm   = entry[f'{split}_cm'].astype(float)
                cm_n = cm / cm.sum(axis=1, keepdims=True)
                ax   = axes[i] if n > 1 else axes
                im   = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)

                for r in range(cm.shape[0]):
                    for c in range(cm.shape[1]):
                        color = 'white' if cm_n[r, c] > 0.6 else 'black'
                        ax.text(c, r,
                                f'{int(cm[r,c])}\n({cm_n[r,c]:.0%})',
                                ha='center', va='center',
                                fontsize=9, color=color)

                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(['Pred Neg', 'Pred Pos'], fontsize=8)
                ax.set_yticklabels(['Real Neg', 'Real Pos'], fontsize=8)
                ax.set_title(f'Fold {entry["fold"]}', fontsize=9)

            plt.colorbar(im, ax=axes if n > 1 else axes,
                         fraction=0.046, pad=0.04,
                         label='Recall normalizado')
            plt.tight_layout()
            plt.savefig(
                os.path.join(DIR_FIGURES,
                             f'confusion_{split}_{TASK}_{MODEL_NAME}.png'),
                dpi=150, bbox_inches='tight'
            )
            plt.close()
        print(f'  [Fig] Confusion matrices (val + test) salvas.')

    # ── Histórico de treinamento ───────────────────────────────
    def _plot_training_histories(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f'Histórico de treinamento — {TASK} ({MODEL_NAME})', fontsize=12
        )

        for i, h in enumerate(self._histories):
            ep    = range(1, len(h['train_loss']) + 1)
            color = FOLD_COLORS[i % len(FOLD_COLORS)]
            label = f'Fold {i+1}'
            axes[0].plot(ep, h['train_loss'], '--', color=color,
                         alpha=0.7, label=f'{label} train')
            axes[0].plot(ep, h['val_loss'],   '-',  color=color,
                         alpha=0.9, label=f'{label} val')
            axes[1].plot(ep, h['auc'], color=color, label=label)
            axes[2].plot(ep, h['f1'],  color=color, label=label)

        for ax, title, ylabel in zip(
            axes,
            ['Loss (treino vs validação)', 'AUC-ROC (validação)', 'F1-Score (validação)'],
            ['Loss', 'AUC', 'F1']
        ):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Época')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(DIR_FIGURES,
                         f'training_history_{TASK}_{MODEL_NAME}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f'  [Fig] Histórico de treinamento salvo.')

    # ── Scree plot comparativo ─────────────────────────────────
    def _plot_scree_comparative(self):
        fig, ax = plt.subplots(figsize=(9, 4))

        for entry in self._scree_data:
            cumul = np.cumsum(entry['explained']) * 100
            ax.plot(range(1, len(cumul)+1), cumul,
                    color=FOLD_COLORS[(entry['fold']-1) % len(FOLD_COLORS)],
                    label=f'Fold {entry["fold"]}', linewidth=1.5)

        ax.axhline(90, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        ax.text(len(self._scree_data[0]['explained']) * 0.97, 91,
                '90%', color='gray', fontsize=9, ha='right')
        ax.set_xlabel('Número de componentes principais')
        ax.set_ylabel('Variância acumulada (%)')
        ax.set_title(f'Scree plot comparativo — {TASK} ({MODEL_NAME})')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(DIR_PCA_FIG,
                         f'scree_comparative_{TASK}_{MODEL_NAME}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f'  [Fig] Scree comparativo salvo.')

    # ── PCA scatter comparativo ────────────────────────────────
    def _plot_pca_comparative(self):
        n    = len(self._pca_data)
        cols = min(n, 3)
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        axes = np.array(axes).flatten()
        fig.suptitle(
            f'PCA — PC1 vs PC2 por fold — {TASK} ({MODEL_NAME})', fontsize=12
        )

        for i, entry in enumerate(self._pca_data):
            ax     = axes[i]
            f2d    = entry['features_2d']
            labels = entry['labels']

            for cls, color, lbl in [(1, CLASS_COLORS['pos'], 'Positivo'),
                                     (0, CLASS_COLORS['neg'], 'Negativo')]:
                pts = f2d[labels == cls]
                ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.5,
                           s=15, label=lbl, zorder=2)

                if len(pts) > 2:
                    mean = pts.mean(axis=0)
                    cov  = np.cov(pts.T)
                    ev, evec = np.linalg.eigh(cov)
                    order = ev.argsort()[::-1]
                    ev, evec = ev[order], evec[:, order]
                    angle = np.degrees(np.arctan2(*evec[:, 0][::-1]))
                    ell = Ellipse(xy=mean,
                                  width=2*np.sqrt(max(ev[0], 0)),
                                  height=2*np.sqrt(max(ev[1], 0)),
                                  angle=angle, edgecolor=color,
                                  fc='None', lw=1.5,
                                  linestyle='--', alpha=0.7)
                    ax.add_patch(ell)

            ax.set_title(f'Fold {entry["fold"]}', fontsize=10)
            ax.set_xlabel('PC1', fontsize=8)
            ax.set_ylabel('PC2', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(DIR_PCA_FIG,
                         f'pca_comparative_{TASK}_{MODEL_NAME}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f'  [Fig] PCA comparativo salvo.')

    # ── Pseudotempo comparativo ────────────────────────────────
    def _plot_pseudotime_comparative(self):
        n    = len(self._pc_data)
        cols = min(n, 3)
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows * 2, cols,
                                 figsize=(5 * cols, 4 * rows * 2))
        axes = axes.reshape(rows * 2, cols)
        fig.suptitle(
            f'Pseudotempo (arclength) por fold — {TASK} ({MODEL_NAME})',
            fontsize=12
        )

        for i, (pc_entry, pca_entry) in enumerate(
            zip(self._pc_data, self._pca_data)
        ):
            row_hist = (i // cols) * 2
            col      = i % cols
            proj     = pc_entry['projections']
            labels   = pca_entry['labels']
            fold     = pc_entry['fold']

            ax_hist = axes[row_hist][col]
            ax_box  = axes[row_hist + 1][col]

            # Histograma
            for cls, color, lbl in [(1, CLASS_COLORS['pos'], 'Positivo'),
                                     (0, CLASS_COLORS['neg'], 'Negativo')]:
                vals = proj[labels == cls]
                ax_hist.hist(vals, bins=18, alpha=0.55, color=color,
                             label=f'{lbl} (n={len(vals)})', density=True)

            _, pval = mannwhitneyu(proj[labels==1], proj[labels==0],
                                   alternative='two-sided')
            ax_hist.set_title(f'Fold {fold} — Mann-Whitney p={pval:.4f}',
                              fontsize=9)
            ax_hist.set_xlabel('Pseudotempo')
            ax_hist.set_ylabel('Densidade')
            ax_hist.legend(fontsize=7)
            ax_hist.grid(alpha=0.2)

            # Boxplot
            bp = ax_box.boxplot([proj[labels==1], proj[labels==0]],
                                patch_artist=True,
                                medianprops=dict(color='black', linewidth=1.5))
            for patch, color in zip(bp['boxes'],
                                    [CLASS_COLORS['pos'], CLASS_COLORS['neg']]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax_box.set_xticks([1, 2])
            ax_box.set_xticklabels(['Positivo', 'Negativo'], fontsize=8)
            ax_box.set_ylabel('Pseudotempo')
            ax_box.grid(axis='y', alpha=0.2)

        # Desativar subplots vazios
        for j in range(n * 2, rows * 2 * cols):
            r = j // cols
            c = j % cols
            if r < rows * 2:
                axes[r][c].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(DIR_PCA_FIG,
                         f'pseudotime_comparative_{TASK}_{MODEL_NAME}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f'  [Fig] Pseudotempo comparativo salvo.')

    # ── Emissões de carbono ────────────────────────────────────
    def _plot_emissions(self):
        folds = [e['fold']   for e in self._emissions]
        kgs   = [e['kg_co2'] for e in self._emissions]
        total = sum(kgs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(
            f'Emissões de carbono — {TASK} ({MODEL_NAME})', fontsize=12
        )

        bars = ax1.bar([f'Fold {f}' for f in folds], kgs,
                       color=FOLD_COLORS[:len(folds)], alpha=0.8)
        for bar, kg in zip(bars, kgs):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(kgs) * 0.01,
                     f'{kg*1000:.1f}g', ha='center', va='bottom', fontsize=9)
        ax1.set_ylabel('kg CO₂eq')
        ax1.set_title('Por fold')
        ax1.grid(axis='y', alpha=0.3)

        acumulado = np.cumsum(kgs)
        ax2.plot(range(len(folds)), acumulado,
                 marker='o', color='#4C72B0', linewidth=2)
        ax2.fill_between(range(len(folds)), acumulado,
                         alpha=0.15, color='#4C72B0')
        ax2.set_xticks(range(len(folds)))
        ax2.set_xticklabels([f'Fold {f}' for f in folds])
        ax2.set_ylabel('kg CO₂eq acumulado')
        ax2.set_title(f'Acumulado — Total: {total*1000:.1f}g CO₂eq')
        ax2.grid(alpha=0.3)

        km_carro  = total / 0.00021   # ~210g CO2/km
        streaming = total / 0.036     # ~36g CO2/hora Netflix
        ax2.text(0.02, 0.95,
                 f'≈ {km_carro:.2f} km de carro\n≈ {streaming:.1f}h de streaming',
                 transform=ax2.transAxes, fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

        plt.tight_layout()
        plt.savefig(
            os.path.join(DIR_FIGURES,
                         f'emissions_{TASK}_{MODEL_NAME}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()

        # CSV de emissões
        df_em = pd.DataFrame(self._emissions)
        df_em['kg_co2_acumulado'] = np.cumsum(df_em['kg_co2'])
        df_em['g_co2']            = df_em['kg_co2'] * 1000
        df_em.loc[len(df_em)]     = {
            'fold': 'TOTAL', 'kg_co2': total,
            'kg_co2_acumulado': total, 'g_co2': total * 1000
        }
        df_em.to_csv(
            os.path.join(DIR_METRICS,
                         f'emissions_{TASK}_{MODEL_NAME}.csv'), index=False
        )
        print(f'  [Fig+CSV] Emissões salvos.')

# ═══════════════════════════════════════════════════════════════
# 13. Cross-validation — orquestra todo o pipeline
# ═══════════════════════════════════════════════════════════════
def cross_validation():
    """
    Pipeline principal com 5-fold StratifiedGroupKFold.

    Fluxo por fold:
        1. Treino + early stopping (EmissionsTracker mede só o treino)
        2. Avaliação val + test interno (métricas de classificação)
        3. Extração de features via hook no avgpool
        4. PCA (StandardScaler + PCA 50 componentes)
        5. Principal Curve no espaço PC1-PC2
        6. Métricas geométricas
        7. log_fold no ReportManager

    Ao final:
        reporter.finalize() gera todos os CSVs e figuras consolidadas.
    """
    dataset  = DFUDataset("../data/ischaemia", transform=transform)
    skf      = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    reporter = ReportManager()

    for fold, (train_idx, val_idx) in enumerate(skf.split(
        X=np.zeros(len(dataset.labels)),
        y=dataset.labels,
        groups=dataset.identifiers
    )):
        print(f'\n{"="*60}\n  FOLD {fold+1}/5\n{"="*60}')

        train_idx, test_idx = train_test_split(
            train_idx,
            test_size=0.2,
            stratify=np.array(dataset.labels)[train_idx],
            random_state=42
        )

        train_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx)
        )
        test_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx)
        )
        val_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(val_idx)
        )

        # ── Treinamento (emissões medidas só aqui) ───────────
        model     = create_model()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

        tracker = codecarbon.EmissionsTracker(
            output_dir=DIR_EMISSIONS,
            output_file=f"fold{fold+1}_train_emissions.csv",
            log_level="error"   # suprime logs do codecarbon no terminal
        )
        tracker.start()
        history = train_model(model, train_loader, val_loader,
                              criterion, optimizer, fold+1)
        emissions_kg = tracker.stop() or 0.0

        # ── Avaliação de classificação ───────────────────────
        model.load_state_dict(torch.load(f'best_model_fold{fold+1}.pth'))
        _, test_metrics = evaluate_model(model, test_loader, criterion)
        _, val_metrics  = evaluate_model(model, val_loader,  criterion)

        # ── Extração de features ─────────────────────────────
        # Usa todos os índices do fold para o mapa completo do espaço latente
        full_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(
                np.concatenate([train_idx, val_idx, test_idx])
            )
        )
        features_np, feat_labels = extract_features(model, full_loader)
        print(f'  [Fold {fold+1}] Features: {features_np.shape}')

        # ── PCA ──────────────────────────────────────────────
        features_pca, _, _, explained, cumulative = run_pca(features_np)
        n_for_90 = int(np.searchsorted(cumulative, 0.90)) + 1
        print(f'  [Fold {fold+1}] PCs para 90% variância: {n_for_90}')

        # Salva features PCA (10 PCs) por fold para análises externas
        pca_df = pd.DataFrame(features_pca[:, :10],
                              columns=[f'PC{i+1}' for i in range(10)])
        pca_df['label'] = feat_labels
        pca_df['fold']  = fold + 1
        pca_df.to_csv(
            os.path.join(DIR_METRICS, f'pca_features_fold{fold+1}.csv'),
            index=False
        )

        # ── Principal Curves ─────────────────────────────────
        pc_obj, projections, distances, pc_metrics = fit_principal_curve(
            features_pca[:, :2], feat_labels, fold+1
        )

        # ── Métricas geométricas ─────────────────────────────
        geo = compute_geometric_metrics(
            features_pca[:, :2], feat_labels, projections, distances
        )
        geo.update(pc_metrics)
        geo['n_pcs_for_90pct'] = n_for_90

        print(f'  [Fold {fold+1}] Silhouette: {geo["silhouette"]:.4f} | '
              f'Centroid dist: {geo["centroid_distance_pca"]:.4f} | '
              f'Pseudotime p: {geo["pseudotime_mannwhitney_pval"]:.4f} | '
              f'CO₂: {emissions_kg*1000:.2f}g')

        # ── Registra no reporter ─────────────────────────────
        reporter.log_fold(
            fold         = fold + 1,
            val_metrics  = val_metrics,
            test_metrics = test_metrics,
            geo_metrics  = geo,
            history      = history,
            features_2d  = features_pca[:, :2],
            feat_labels  = feat_labels,
            projections  = projections,
            distances    = distances,
            explained    = explained,
            emissions_kg = emissions_kg
        )

    # ── Gera todas as saídas consolidadas ────────────────────
    reporter.finalize()

# ═══════════════════════════════════════════════════════════════
# Ponto de entrada
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cross_validation()
