import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_conceptual_framework():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Hide axes
    ax.axis('off')
    
    # --- Left: Yogacara / Human Mind ---
    ax.text(0.25, 0.95, "Yogacara Phenomenology\n(Human Mind)", ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Box 1: Paratantra (Sensory)
    rect_para = patches.Rectangle((0.1, 0.1), 0.3, 0.2, linewidth=2, edgecolor='black', facecolor='#e6f3ff')
    ax.add_patch(rect_para)
    ax.text(0.25, 0.2, "Paratantra\n(Dependent Arising)\nSensory Input", ha='center', va='center', fontsize=10)
    
    # Box 2: Parikalpita (Priors)
    rect_pari = patches.Rectangle((0.1, 0.6), 0.3, 0.2, linewidth=2, edgecolor='black', facecolor='#ffe6e6')
    ax.add_patch(rect_pari)
    ax.text(0.25, 0.7, "Parikalpita\n(Mental Construction)\nRigid Priors / Attachment", ha='center', va='center', fontsize=10)
    
    # Box 3: Vipassana (Metacognition)
    rect_vip = patches.Rectangle((0.1, 0.35), 0.3, 0.2, linewidth=2, edgecolor='black', facecolor='#e6ffe6', linestyle='--')
    ax.add_patch(rect_vip)
    ax.text(0.25, 0.45, "Vipassana / Insight\n(Metacognitive Monitoring)", ha='center', va='center', fontsize=10, fontstyle='italic')

    # Arrows Left
    ax.annotate("", xy=(0.25, 0.6), xytext=(0.25, 0.55), arrowprops=dict(arrowstyle="->", lw=1.5)) # Vip -> Pari
    ax.annotate("", xy=(0.25, 0.35), xytext=(0.25, 0.3), arrowprops=dict(arrowstyle="->", lw=1.5)) # Vip -> Para (Observation)
    
    # --- Right: PCCF / AI Architecture ---
    ax.text(0.75, 0.95, "PCCF Architecture\n(AI System)", ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Box 4: Sensory Input (o_t)
    rect_input = patches.Rectangle((0.6, 0.1), 0.3, 0.2, linewidth=2, edgecolor='black', facecolor='#e6f3ff')
    ax.add_patch(rect_input)
    ax.text(0.75, 0.2, "Sensory Input ($o_t$)\nPrediction Error Source", ha='center', va='center', fontsize=10)
    
    # Box 5: Internal State (mu)
    rect_state = patches.Rectangle((0.6, 0.6), 0.3, 0.2, linewidth=2, edgecolor='black', facecolor='#ffe6e6')
    ax.add_patch(rect_state)
    ax.text(0.75, 0.7, "Internal State ($g(\mu_t)$)\nTop-down Prediction", ha='center', va='center', fontsize=10)
    
    # Box 6: PCCF Controller
    rect_pccf = patches.Rectangle((0.6, 0.35), 0.3, 0.2, linewidth=2, edgecolor='red', facecolor='#e6ffe6')
    ax.add_patch(rect_pccf)
    ax.text(0.75, 0.45, "**PCCF Controller**\nPrecision-Weighted Gain ($K_t$)\n$\tanh(Error^2 / \sigma^2)$", ha='center', va='center', fontsize=10, color='red')

    # Arrows Right (Computational Flow)
    # Bottom-up Error
    ax.annotate("Prediction Error ($e_t$)", xy=(0.75, 0.35), xytext=(0.75, 0.3), arrowprops=dict(arrowstyle="->", lw=2, color='blue'), ha='center', fontsize=8)
    
    # Modulation
    ax.annotate("Gain Modulation ($K_t$)", xy=(0.75, 0.6), xytext=(0.75, 0.55), arrowprops=dict(arrowstyle="->", lw=2, color='red'), ha='center', fontsize=8)

    # --- Mapping Lines (Dashed) ---
    # Paratantra <-> Input
    ax.annotate("", xy=(0.6, 0.2), xytext=(0.4, 0.2), arrowprops=dict(arrowstyle="<->", linestyle="--", color='gray'))
    # Parikalpita <-> State
    ax.annotate("", xy=(0.6, 0.7), xytext=(0.4, 0.7), arrowprops=dict(arrowstyle="<->", linestyle="--", color='gray'))
    # Vipassana <-> PCCF
    ax.annotate("Isomorphic Mapping", xy=(0.6, 0.45), xytext=(0.4, 0.45), arrowprops=dict(arrowstyle="<->", linestyle="--", color='purple', lw=2), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('pccf_conceptual_framework.png', dpi=300, bbox_inches='tight')
    print("Figure saved to pccf_conceptual_framework.png")

if __name__ == "__main__":
    draw_conceptual_framework()
