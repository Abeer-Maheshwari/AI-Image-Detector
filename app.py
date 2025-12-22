import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path
from collections import Counter
import threading
import time

# Import your model
from CNN_train import CNN, DEVICE

# GradCAM class (unchanged)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, class_idx):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activation, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        return heatmap

# Load model bundle
WEIGHTS_PATH = Path('BeyondAI_Extension/trained_models.pth')
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError("trained_models.pth not found!")

bundle = torch.load(WEIGHTS_PATH, map_location=DEVICE)
optimizers = list(bundle['optimizers'].keys())
classes = ['Real', 'AI']

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BeyondAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BeyondAI - AI vs Real Image Detector")
        self.root.configure(bg="#0f0f1a")
        self.root.geometry("1300x900")

        # Title with glow
        title = tk.Label(root, text="BeyondAI Detector", font=("Helvetica", 28, "bold"),
                         fg="#00ffff", bg="#0f0f1a")
        title.pack(pady=30)

        # Upload button with hover effect
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.start_analysis,
                                    font=("Helvetica", 16, "bold"), bg="#0088aa", fg="white",
                                    activebackground="#00ccff", activeforeground="white",
                                    relief="flat", padx=30, pady=15, cursor="hand2", bd=0)
        self.upload_btn.pack(pady=20)
        self.upload_btn.bind("<Enter>", lambda e: self.upload_btn.config(bg="#00aadd"))
        self.upload_btn.bind("<Leave>", lambda e: self.upload_btn.config(bg="#0088aa"))

        # Original image
        self.orig_label = tk.Label(root, bg="#1a1a2e", relief="solid", bd=3)
        self.orig_label.pack(pady=20)

        # Canvas for scrollable results
        canvas = tk.Canvas(root, bg="#0f0f1a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg="#0f0f1a")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=30, pady=10)
        scrollbar.pack(side="right", fill="y")

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Status
        self.status = tk.Label(root, text="Ready", fg="#00ccaa", bg="#0f0f1a", font=("Helvetica", 12))
        self.status.pack(pady=10)

        self.cards = []  # To store card frames for animation

    def fade_in(self, widget, alpha=0):
        if alpha >= 1.0:
            return
        alpha += 0.08
        widget.config(bg=f"#{int(30*alpha):02x}{int(40*alpha):02x}{int(60*alpha):02x}")
        widget.after(30, lambda: self.fade_in(widget, alpha))

    def pulse_glow(self, label, phase=0):
        colors = ["#00ff88", "#00ffcc", "#00ffff", "#00ccff", "#00aaff"]
        color = colors[phase % len(colors)]
        label.config(fg=color)
        label.after(800, lambda: self.pulse_glow(label, phase + 1))

    def hover_glow(self, frame, on_enter=False):
        if on_enter:
            frame.config(relief="raised", bd=6, bg="#1a2a3a")
        else:
            frame.config(relief="raised", bd=3, bg="#2d2d2d")

    def start_analysis(self):
        threading.Thread(target=self.upload_image, daemon=True).start()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        self.root.after(0, lambda: self.status.config(text="Processing image...", fg="#ffaa00"))

        try:
            raw_img = Image.open(file_path).convert('RGB')
            input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

            # Display original with subtle enhancement
            self.root.after(0, lambda: self.status.config(text="Displaying original..."))
            display_img = raw_img.copy()
            display_img.thumbnail((450, 450), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_img)
            self.root.after(0, lambda: self.orig_label.config(image=photo, text=""))
            self.root.after(0, lambda: setattr(self.orig_label, 'image', photo))

            # Clear previous
            self.root.after(0, lambda: [w.destroy() for w in self.scrollable_frame.winfo_children()])
            self.cards = []

            results = []
            all_preds = []

            total = len(optimizers)
            for idx, opt_name in enumerate(optimizers):
                self.root.after(0, lambda t=idx+1: self.status.config(text=f"Analyzing with {opt_name}... ({t}/{total})"))

                state_dict = bundle['optimizers'][opt_name]['wd']
                model = CNN().to(DEVICE)
                model.load_state_dict(state_dict)
                model.eval()

                target_layer = model.features[6]
                grad_cam = GradCAM(model, target_layer)

                output = model(input_tensor)
                probs = F.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item() * 100
                pred_label = classes[pred_idx]
                all_preds.append(pred_label)

                model.zero_grad()
                output[0, pred_idx].backward()
                heatmap = grad_cam.generate_heatmap(pred_idx)

                heatmap = cv2.resize(heatmap, (raw_img.width, raw_img.height))
                heatmap = np.uint8(255 * heatmap)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                overlay_pil = Image.fromarray(overlay)
                overlay_pil.thumbnail((280, 280), Image.Resampling.LANCZOS)

                results.append((overlay_pil, opt_name, pred_label, confidence))

            # Ensemble
            vote = Counter(all_preds)
            final_pred = vote.most_common(1)[0][0]
            agreement = (vote[final_pred] / total) * 100

            self.root.after(0, lambda: self.status.config(text="Rendering results..."))

            # Ensemble result with pulsing glow
            ensemble_label = tk.Label(self.scrollable_frame,
                                      text=f"FINAL VERDICT: {final_pred.upper()}\n{agreement:.0f}% MODEL AGREEMENT",
                                      font=("Helvetica", 22, "bold"),
                                      fg="#00ff88" if final_pred == "Real" else "#ff4444",
                                      bg="#0f0f1a")
            ensemble_label.grid(row=0, column=0, columnspan=4, pady=40)
            self.root.after(100, lambda: self.pulse_glow(ensemble_label))

            # Animate cards in one by one
            for i, (img, opt_name, pred, conf) in enumerate(results):
                self.root.after(200 + i * 150, lambda img=img, opt=opt_name, prd=pred, cnf=conf, idx=i: self.create_card(img, opt, prd, cnf, idx))

            self.root.after(0, lambda: self.status.config(text="Analysis Complete! ✨", fg="#00ffff"))

        except Exception as e:
            self.root.after(0, lambda: self.status.config(text=f"Error: {str(e)}", fg="#ff4444"))
            print(e)

    def create_card(self, img, opt_name, pred, conf, index):
        row = (index // 4) + 2
        col = index % 4

        frame = tk.Frame(self.scrollable_frame, bg="#1a1a2e", relief="raised", bd=3, padx=10, pady=10)
        frame.grid(row=row, column=col, padx=20, pady=20, sticky="n")

        # Hover effect
        frame.bind("<Enter>", lambda e: self.hover_glow(frame, True))
        frame.bind("<Leave>", lambda e: self.hover_glow(frame, False))

        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(frame, image=photo, bg="#1a1a2e")
        img_label.image = photo
        img_label.pack(pady=10)

        tk.Label(frame, text=opt_name, font=("Helvetica", 14, "bold"), fg="#00ccff", bg="#1a1a2e").pack()
        color = "#00ff88" if pred == "Real" else "#ff4444"
        tk.Label(frame, text=pred, font=("Helvetica", 20, "bold"), fg=color, bg="#1a1a2e").pack(pady=5)
        tk.Label(frame, text=f"{conf:.1f}% confidence", font=("Helvetica", 11), fg="#aaaaaa", bg="#1a1a2e").pack()

        # Fade-in animation
        frame.config(bg="#000000")
        self.fade_in(frame)

        self.cards.append(frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = BeyondAIApp(root)
    root.mainloop()