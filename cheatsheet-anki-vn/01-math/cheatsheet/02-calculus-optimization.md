# Cheatsheet — Calculus & Optimization cho AI

> Phase 1 · Sheet 02 · Phase Toán
> Hướng dạng: trực giác → công thức → code numpy/torch → pitfall → tradeoff. Tiếng Việt + giữ thuật ngữ TA gốc.

---

## 0. Roadmap sheet này

1. Derivative (đạo hàm) — slope tiếp tuyến
2. Partial derivative (đạo hàm riêng) — slope theo 1 biến
3. Gradient — vector đạo hàm riêng (chỉ đạo đi lên dốc nhanh nhất)
4. Jacobian & Hessian — tổng quát ma trận của đạo hàm
5. Chain rule — nền tảng backpropagation
6. Taylor series — xấp xỉ cục bộ, nền Newton's method
7. Convexity — bảo đảm cực tiểu toàn cục
8. Lagrangian — ràng buộc → bài toán KKT
9. Gradient Descent (GD / SGD / Mini-batch SGD)
10. Momentum / Nesterov
11. AdaGrad / RMSProp / Adam / AdamW / LAMB
12. LR schedule (warmup, cosine, exponential)
13. Gradient clipping / weight decay / EMA
14. Second-order methods (Newton, BFGS, L-BFGS) — khi nào dùng
15. Stochastic methods (SGD with noise, simulated annealing — nhanh)

---

## 1. Derivative (Đạo hàm) — slope tiếp tuyến

### Định nghĩa
Đạo hàm của \(f: \mathbb{R} \to \mathbb{R}\) tại điểm \(x\):
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

- Hệ số góc của tiếp tuyến tại \(x\)
- Dương nếu \(f\) tăng, âm nếu giảm, 0 nếu cực trị
- Viết gọn: \(\frac{df}{dx}\), \(f'(x)\), \(\dot f\) (theo thời gian)

### Quy tắc cơ bản
| Quy tắc | Công thức |
|---------|-----------|
| Tổng/hiệu | \((f+g)' = f' + g'\) |
| Tích | \((fg)' = f'g + fg'\) |
| Thương | \((f/g)' = \frac{f'g - fg'}{g^2}\) |
| Dây chuyền | \((f \circ g)'(x) = f'(g(x)) \cdot g'(x)\) |
| Lũy thừa | \((x^n)' = n x^{n-1}\) |
| Exponential | \((e^x)' = e^x\) |
| Log | \((\ln x)' = 1/x\) |
| Sin/Cos | \((\sin x)' = \cos x\), \((\cos x)' = -\sin x\) |

### Đạo hàm thường gặp trong AI
| Hàm | Đạo hàm |
|-----|---------|
| Sigmoid \(\sigma(x) = \frac{1}{1+e^{-x}}\) | \(\sigma'(x) = \sigma(x)(1 - \sigma(x))\) |
| Tanh \(\tanh(x)\) | \(1 - \tanh^2(x)\) |
| ReLU \(\max(0, x)\) | 0 nếu \(x < 0\), 1 nếu \(x > 0\), không xác định tại \(x=0\) |
| Leaky-ReLU \(\max(ax, x)\) với \(a=0.01\) | \(a\) nếu \(x<0\), 1 nếu \(x>0\) |
| Softmax \(\text{softmax}(z)_i\) | xem công thức Jacobian §4 |
| \(\ln(1+e^x)\) (softplus) | \(\sigma(x)\) |
| \(x \ln x\) | \(\ln x + 1\) |

### Numerical derivative (an ninh khi check autograd)
```python
import numpy as np

def num_grad(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)   # trung tâm, sai số O(h^2)

# Check gradient của hàm mất mát
assert np.allclose(num_grad(loss, x), autograd_grad), "Gradient sai!"
```

### Pitfall
- **`h` quá nhỏ** (1e-20) → floating point cancelllation, đạo hàm số bằng 0. Dùng 1e-5 đến 1e-7 cho float64, 1e-3 cho float32.
- **Đạo hàm ReLU tại 0** không xác định → thực tế gán giá trị 0 (subgradient) trong PyTorch/TF.
- **Cast 1 số int sang float** khi tính đạo hàm trong numpy: `1 / x` integer division nếu `x` int.

---

## 2. Partial Derivative (Đạo hàm riêng) — slope theo 1 biến

Cho hàm \(f: \mathbb{R}^n \to \mathbb{R}\):
$$\frac{\partial f}{\partial x_i}(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h \mathbf{e}_i) - f(\mathbf{x})}{h}$$

- Đạo hàm theo 1 biến, **giữ các biến khác cố định**
- Ký hiệu: \(\frac{\partial f}{\partial x_i}\), \(f_{x_i}\), \(\partial_i f\)

### Ví dụ
\(f(x, y) = x^2 y + \sin(xy)\):
- \(\frac{\partial f}{\partial x} = 2xy + y \cos(xy)\)
- \(\frac{\partial f}{\partial y} = x^2 + x \cos(xy)\)

### Code numpy
```python
import numpy as np

def partial(f, x, i, h=1e-5):
    e = np.zeros_like(x, dtype=float)
    e[i] = h
    return (f(x + e) - f(x - e)) / (2 * h)

# f: R^3 -> R
f = lambda v: v[0]**2 * v[1] + np.sin(v[0]*v[2])
x = np.array([1.0, 2.0, 3.0])
print(partial(f, x, 0))   # df/dx
```

---

## 3. Gradient — vector đạo hàm riêng

Gradient của \(f: \mathbb{R}^n \to \mathbb{R}\):
$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### Tính chất cực kỳ quan trọng
1. **\(\nabla f\) chỉ đạo dốc đi lên nhanh nhất** (steepest ascent direction)
2. **\(-\nabla f\) chỉ đạo dốc đi xuống nhanh nhất** (steepest descent) → nền Gradient Descent
3. \(\nabla f\) **trực giao với đường mức (level set)** của \(f\)
4. Cực trị: \(\nabla f(\mathbf{x}^*) = \mathbf{0}\) (điểm dừng), kết hợp Hessian để biết min/max/saddle

### Code numpy
```python
import numpy as np

def grad(f, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g

# Test: f(x,y,z) = x^2 + y^2 + z^2  → grad = (2x, 2y, 2z)
f = lambda v: np.sum(v**2)
print(grad(f, np.array([1.0, 2.0, 3.0])))   # ≈ [2, 4, 6]
```

### Autograd với PyTorch (production)
```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()       # forward
y.backward()              # backward — tính đạo hàm
print(x.grad)            # tensor([2., 4., 6.])
```

### Pitfall
- **Quên `requires_grad=True`** → `x.grad` là None
- **Quên `x.grad.zero_()` hay `opt.zero_grad()`** trước backward lần sau → gradient cộng dồn (PyTorch mặc định accumulate)
- **`y.backward()` 2 lần trên cùng graph** → lỗi (graph đã freed). Phải `retain_graph=True` hoặc re-build forward.

---

## 4. Jacobian & Hessian — tổng quát hóa ma trận

### Jacobian (cho vector-valued)
Cho \(f: \mathbb{R}^n \to \mathbb{R}^m\), Jacobian là ma trận \(m \times n\):
$$\mathbf{J}_f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

- Mỗi **hàng** là gradient của \(f_i\) (output thứ \(i\))
- Cho biết biến đổi tuyến tính cục bộ của \(f\)
- Determinant của Jacobian → hệ số co giãn thể tích (dùng trong normalizing flow)

### Hessian (cho scalar-valued)
Cho \(f: \mathbb{R}^n \to \mathbb{R}\), Hessian là ma trận vuông \(n \times n\):
$$\mathbf{H}_f(\mathbf{x}) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \end{bmatrix}$$

- Ma trận đối xứng nếu \(f\) có đạo hàm riêng liên tục (Schwarz)
- **Eigenvalue của Hessian + critical point**:
  - Tất cả dương \(\Rightarrow\) local minimum
  - Tất cả âm \(\Rightarrow\) local maximum
  - Vừa dương vừa âm \(\Rightarrow\) saddle point (yên ngựa)
  - Có zero \(\Rightarrow\) cần test bậc cao hơn

### Softmax Jacobian (cực kỳ quan trọng cho cross-entropy)
\(\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}\). Jacobian:
$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i (1 - p_i) & i = j \\ -p_i p_j & i \neq j \end{cases}$$

Viết gọn ma trận: \(\mathbf{J} = \text{diag}(\mathbf{p}) - \mathbf{p} \mathbf{p}^T\)

### Code numpy
```python
import numpy as np

def jacobian(f, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    y0 = np.atleast_1d(f(x))
    n, m = x.size, y0.size
    J = np.zeros((m, n))
    for i in range(n):
        e = np.zeros_like(x); e[i] = h
        J[:, i] = (f(x + e) - f(x - e)) / (2 * h)
    return J

def hessian(f, x, h=1e-4):
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(n):
            ei = np.zeros_like(x); ei[i] = h
            ej = np.zeros_like(x); ej[j] = h
            H[i, j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*h*h)
    return H
```

### Dùng trong phân tích
- **Newton's method**: \(\mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{H}^{-1} \nabla f\) — bậc 2 hội tụ
- **Conditioning**: \(\kappa(\mathbf{H}) = \frac{\lambda_{\max}}{\lambda_{\min}}\) → càng cao, GD càng chậm (elongated valley)

---

## 5. Chain Rule — nền tảng Backpropagation

### Dạng 1 biến
\((f \circ g)'(x) = f'(g(x)) \cdot g'(x)\)

### Dạng vector (cho mạng nơ-ron)
Nếu \(\mathbf{z} = f(\mathbf{y})\) và \(\mathbf{y} = g(\mathbf{x})\):
$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{J}_f \cdot \mathbf{J}_g$$

### Mở rộng: multilayer
Cho \(\ell = L \circ \sigma \circ W \circ \cdots \circ \sigma \circ W_1(x)\):
$$\frac{\partial \ell}{\partial W_1} = \frac{\partial \ell}{\partial L} \cdot \frac{\partial L}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial W} \cdots$$

### Backpropagation = chain rule + memoization
- Forward pass: tính từng layer, **lưu lại activations**
- Backward pass: tính gradient ngược từ output về input, **tái dùng activations đã lưu**
- Độ phức tạp: \(O(\text{forward})\), không phải \(O(\text{forward}^2)\)

### Pitfall
- **Quên activation function không khả vi tại 0** (ReLU) → subgradient, vẫn chạy được
- **Dying ReLU**: gradient về 0 khi input luôn âm → neuron "chết", không học. Fix bằng Leaky-ReLU/GELU/SwiGLU.
- **Bursting gradient / vanishing gradient**: mạng sâu × đạo hàm < 1 → gradient nhân dồn → 0 (RNN đặc biệt). Fix: LSTM/GRU/Residual/attention.

### Code numpy (backprop tay)
```python
import numpy as np

# Mang 1 layer: y = sigmoid(W @ x + b), loss = (y - target)^2
x = np.array([1.0, 2.0])
W = np.array([[0.1, 0.2], [0.3, 0.4]])
b = np.array([0.0, 0.0])
target = np.array([1.0, 0.0])

# Forward
z = W @ x + b
y = 1 / (1 + np.exp(-z))
loss = np.sum((y - target)**2)

# Backward
dy = 2 * (y - target)                          # dLoss/dy
dz = dy * y * (1 - y)                          # dLoss/dz = dLoss/dy * sigmoid'
dW = np.outer(dz, x)                           # dLoss/dW
db = dz                                        # dLoss/db
print("dW:", dW)
```

---

## 6. Taylor Series — xấp xỉ cục bộ

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!}(x-a)^n$$

- Xấp xỉ bậc 1: \(f(x) \approx f(a) + f'(a)(x-a)\) → tiếp tuyến, cơ sở Gradient Descent
- Xấp xỉ bậc 2: dùng Hessian → Newton's method

### Ứng dụng
- **Newton's method**: giải \(\nabla f = 0\) qua xấp xỉ bậc 2
- **Mô phỏng công thức hàm chuẩn**: \(\sin x \approx x - x^3/6\), \(e^x \approx 1 + x + x^2/2\)
- **Loss xấp xỉ**: GELU xấp xỉ \(\tanh\left(\sqrt{2/\pi}(x + 0.044715 x^3)\right)\)

---

## 7. Convexity — bảo đảm cực tiểu toàn cục

### Định nghĩa
Hàm \(f: \mathbb{R}^n \to \mathbb{R}\) **lồi** (convex) nếu:
$$f(t \mathbf{x} + (1-t) \mathbf{y}) \leq t f(\mathbf{x}) + (1-t) f(\mathbf{y}) \quad \forall t \in [0,1]$$

Trực giác: **dây cung không bao giờ nằm dưới đồ thị**.

### Điều kiện
- \(f\) lồi \(\iff\) Hessian \(\mathbf{H} \succeq 0\) (nửa xác định dương) với mọi x
- \(f\) lồi nghiêm ngặt \(\iff\) \(\mathbf{H} \succ 0\)

### Tính chất cực kỳ quan trọng
- **Mọi cực trị cục bộ = cực trị toàn cục** (không có local min khác global min)
- Gradient Descent hội tụ nếu Learning Rate đủ nhỏ

### Hàm convex hay gặp trong AI
| Loss function | Convex? |
|--------------|---------|
| MSE (LinReg) | ✅ convex |
| Logistic loss (LogReg tuyến tính) | ✅ convex |
| Hinge loss (SVM tuyến tính) | ✅ convex |
| Cross-entropy vs softmax (deep NN) | ❌ không convex (do hidden layer) |
| Mất mát của deep NN | ❌ không convex, nhiều saddle point |

### Tradeoff
- **Loss convex** → dễ tối ưu, hội tụ устойчиво → dùng LinReg / LogReg
- **Loss deep NN không convex** → không bảo đảm toàn cục, nhưng thực tế SGD vẫn tìm được điểm tốt (cực trị cục bộ nhiều nhưng chất lượng tương đương)

---

## 8. Lagrangian — bài toán ràng buộc

Cho bài toán:
$$\min f(\mathbf{x}) \quad \text{sao cho} \quad g_i(\mathbf{x}) \leq 0, \; h_j(\mathbf{x}) = 0$$

Lagrangian:
$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \mu_j h_j(\mathbf{x})$$

with \(\lambda_i \geq 0\). Điều kiện KKT (Karush-Kuhn-Tucker):
1. **Độ dốc bằng 0**: \(\nabla_x \mathcal{L} = 0\)
2. **Khả thi**: \(g_i(\mathbf{x}) \leq 0\), \(h_j(\mathbf{x}) = 0\)
3. **Dual feasibility**: \(\lambda_i \geq 0\)
4. **Complementary slackness**: \(\lambda_i g_i(\mathbf{x}) = 0\)

### Ứng dụng trong AI
- **SVM**: \(\min \frac{1}{2} ||\mathbf{w}||^2\) sao cho \(y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1\) → ra được dual form với \(\alpha_i\)
- **Lagrange dual神经网络 penalty** (regularization): weight decay tương đương ràng buộc \(||\mathbf{w}||^2 \leq t\)
- **Constrained optimization** trong RL (TRPO / PPO dùng trust region constraint)

---

## 9. Gradient Descent — GD / SGD / Mini-batch SGD

### Thuật toán tổng quát
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

### 3 biến thể chính
| Loại | Lấy data |pros | cons |
|------|----------|-----|-------|
| **Gradient Descent (GD)** | Toàn bộ dataset | Hội tụ ổn định | Chậm, RAM lớn, không khả thi cho big data |
| **Stochastic GD (SGD)** | 1 mẫu / bước | Nhanh, though noise | Nhiễu cần LR nhỏ dần |
| **Mini-batch SGD** | batch 32-512 | Cân bằng, song song GPU | Cần tune batch size + LR |

### Pseudo-code (mini-batch SGD)
```
for epoch in epochs:
    shuffle data
    for batch in data.split(batch_size):
        grad = ∇ L_batch(θ)
        θ = θ - η * grad
```

### Code numpy (gradient descent thuần)
```python
import numpy as np

def gd(f, grad_f, x0, lr=0.01, n_iter=1000, tol=1e-6):
    x = x0.copy()
    for i in range(n_iter):
        g = grad_f(x)
        x_new = x - lr * g
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

# Test: minimize f(x) = x^2 + y^2
f = lambda v: v[0]**2 + v[1]**2
g = lambda v: np.array([2*v[0], 2*v[1]])
print(gd(f, g, np.array([5.0, 5.0]), lr=0.1))   # ≈ [0, 0]
```

### PyTorch training loop (production)
```python
import torch
from torch.utils.data import DataLoader

model = MyModel()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(10):
    for X_batch, y_batch in loader:
        opt.zero_grad()                          # 1. Clear gradient
        pred = model(X_batch)                    # 2. Forward
        loss = loss_fn(pred, y_batch)
        loss.backward()                           # 3. Backward (autograd)
        opt.step()                                # 4. Update theta
```

### Pitfall
- **LR quá lớn** → diverge (loss tăng dần, NaN). Bắt đầu 0.001, tăng dần.
- **LR quá nhỏ** → hội tụ chậm, kẹt saddle.
- **Quên `shuffle` data** → epoch đầu học hết class A, sau đó mới class B, gradient direction lệch.
- **Gradient explode** (RNN sâu) → gradient clipping.
- **Gradient vanish** (RNN sâu, sigmoid) → LSTM/GRU/attention.

### Tradeoff batch size
| Batch | LR tương ứng | Đặc điểm |
|-------|-------------|---------|
| 1 (SGD) | nhỏ (1e-4) | Nhiều noise, dễ thoát saddle |
| 64 | 1e-3 | Cân bằng, phổ biến |
| 256-512 | 1e-2 | Tận dụng GPU, gradient "trơn" hơn |
| 4096+ | 5e-2-1e-1 | Cần LR warmup, thường dùng với big batch training (BERT/GPT) |

**Quy tắc tay**:LR scale tuyến tính theo batch size (\(\eta \propto \text{batch}\)) — pause chỉ dùng với warmup, không thì hội tụ kém.

---

## 10. Momentum & Nesterov

### Momentum
$$v_{t+1} = \gamma v_t + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

- \(\gamma \in [0.5, 0.99]\), thông thường 0.9
- Tích lũy gradient → bứt phá qua valley hẹp
- Hình dung: quả bóng lăn xuống dốc có quán tính

### Nesterov Accelerated Gradient (NAG)
$$v_{t+1} = \gamma v_t + \nabla L(\theta_t - \eta \gamma v_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

- Tính gradient tại điểm **dự đoán trước** \(\theta - \eta \gamma v\)
- Hội tụ nhanh hơn momentum thường, ít overshoot

### PyTorch
```python
opt = torch.optim.SGD(params, lr=0.01, momentum=0.9)
opt_nag = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True)
```

---

## 11. Adaptive Optimizers — AdaGrad / RMSProp / Adam / AdamW / LAMB

### AdaGrad
– Tích lũy bình phương gradient cho mỗi param, chia cho sqrt:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t, \quad G_t = \sum_{s=1}^t g_s^2$$

- **Pros**: Tự giảm LR cho feature thưa (NLP xưa)
- **Cons**: \(G_t\) chỉ tăng → LR tiến về 0, ngừng học sớm

### RMSProp (fix AdaGrad)
– \(G_t\)Moving average bình phương:
$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

- \(\beta = 0.9\), \(\epsilon = 1e-8\)
- LR adaptive, không bị decay về 0
- Phổ biến cho RNN

### Adam (default cho DL)
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat v_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

- \(\beta_1 = 0.9\), \(\beta_2 = 0.999\), \(\epsilon = 1e-8\)
- = Momentum + RMSProp + bias correction
- **Default cho DL**, hard to beat out-of-the-box

### AdamW (fix weight decay của Adam)
– Adam + correct weight decay:
$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_t \right)$$

- **Default cho Transformer / LLM fine-tune**
- Khắc phục Adam смеш weight decay với gradient scaling

### LAMB (cho big batch training)
- Layer-wise adaptive moments + trust ratio
- Cho batch 64K+, train BERT/Llama với LR rất cao
- Công thức phức tạp, ít dùng cá nhân

### PyTorch
```python
torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
torch.optim.Adam(params, lr=1e-3)          # legacy
torch.optim.RMSprop(params, lr=1e-3, alpha=0.9)
```

### Bảng chọn optimizer gợi ý
| Task | Optimizer khuyến nghị |
|------|---------------------|
| LinReg / LogReg | SGD với momentum |
| CNN image classification | SGD + momentum 0.9 (generalize tốt) |
| Transformer / LLM | AdamW với weight_decay 0.01-0.1 |
| RNN | RMSProp / Adam |
| Fine-tune LLM (LoRA) | AdamW, weight_decay 0.01, lr 1e-4 đến 5e-4 |
| RL (PPO) | Adam, lr 1e-4 đến 3e-4 |

### Pitfall
- **Adam với weight_decay > 0 dùng AdamW thay Adam + weight_decay** — Adam + decay converge tồi
- **Adam beta2 quá thấp (0.9)** → v_t không smooth, gradient spikes dominate
- **Reset optimizer sau when load checkpoint** thường cần thiết (giữ state OK nếu resume)

---

## 12. LR Schedule

### Warmup + Cosine Decay (LLM chuẩn)
- warmup LR tăng tuyến tính từ 0 → \(LR_{max}\) trong \(T_w\) bước
- Sau warmup, decay theo cosine về \(LR_{min}\):
$$LR(t) = \begin{cases} \frac{t}{T_w} LR_{max} & t < T_w \\ LR_{min} + \frac{1}{2}(LR_{max} - LR_{min})\left(1 + \cos\left(\frac{t - T_w}{T - T_w} \pi\right)\right) & t \geq T_w \end{cases}$$

### Exponential decay
$$LR(t) = LR_0 \cdot \gamma^{t / s}$$
- \(\gamma = 0.96\), \(s = 1000\) bước

### Step decay
- Giảm LR × 0.1 mỗi epoch / giới hạn (PyTorch `StepLR`)

### PyTorch scheduler
```python
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ExponentialLR, LambdaLR, OneCycleLR
)

# Cosine decay đơn giản
sched = CosineAnnealingLR(opt, T_max=30, eta_min=1e-6)
for epoch in range(30):
    train(...)
    sched.step()

# Warmup + cosine (chuẩn LLM) — phải tự viết/LambdaLR
def warmup_cosine(step, warmup=500, total=10000, lr_max=1e-4, lr_min=1e-6):
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(progress * np.pi))

sched = LambdaLR(opt, lr_lambda=lambda s: warmup_cosine(s) / 1e-4)
```

HuggingFace transformers helper:
```python
from transformers import get_cosine_schedule_with_warmup
sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=500, num_training_steps=10000)
```

### Pitfall
- **Quên `scheduler.step()`** → LR không thay đổi
- **Step scheduler theo epoch** nhưng gọi mỗi batch → LR giảm nhanh 32 lần
- **Warmup quá dài** → mất phase học, LR chưa tới max đã decay

---

## 13. Gradient Clipping / Weight Decay / EMA

### Gradient clipping (chống gradient nổ)
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Clip后的梯度norm <= 1.0
```

- Clip theo **norm toàn cục** (ach‬ phổ biến LLM)
- Clip theo **value** (mỗi phần tử): `clip_grad_value_(params, 1.0)` — ít thông dụng hơn

### Weight decay (L2 regularization)
$$L_{total} = L_{data} + \lambda ||\theta||_2^2$$

-使学生 weight nhỏ, chống overfitting
- AdamW tích hợp sẵn

### EMA (Exponential Moving Average của weights)
```python
ema_w = beta * ema_w + (1 - beta) * model_w
```
- Dùng khi inference: thay model.weight bằng ema_w cho kết quả mượt hơn
- Phổ biến GAN diffusion model

---

## 14. Second-order Methods — Newton / BFGS / L-BFGS

### Newton's method
$$\theta_{t+1} = \theta_t - \mathbf{H}^{-1} \nabla f(\theta_t)$$

- Hội tụ bậc 2 (đại số — chỉ vài bước)
- Không cần LR (H scaling sẵn)
- **Cons**: \(\mathbf{H}^{-1}\) \(O(n^3)\), Hessian khổng lồ cho NN (millions params) → không khả thi trong DL

### BFGS / L-BFGS
- BFGS: xấp xỉ inverse Hessian bằng lịch sử gradient
- L-BFGS: limited memory version, không lưu full Hessian
- Hội tụ nhanh hơn first-order cho bài toán convex
- Trong deep learning, chỉ dùng cho **model nhỏ** hoặc **fine-tuning cuối cùng**. Không phải default.

### PyTorch
```python
opt = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)

def closure():
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    return loss

opt.step(closure)  # require closure
```

### Tradeoff
| Phương pháp | Bậc hội tụ | Chi phí / bước | Khi nào dùng |
|------------|-----------|--------------|--------------|
| GD | 1 (linear) | \(O(n)\) | Big data, default |
| SGD / mini-batch | 1 (linear) | \(O(\text{batch})\) | DL production |
| Momentum / Adam | 1 (linear nhưng nhanh hơn) | \(O(n)\) | Default DL |
| Newton | 2 (quadratic) | \(O(n^3)\) | Logistic reg nhỏ |
| L-BFGS | siêu tuyến | \(O(mn)\), m nhỏ | Fine-tune gần optimum |

---

## 15. Pitfalls tổng hợp

| Pitfall | Hiện tượng | Fix |
|---------|-----------|-----|
| NaN loss | LR quá lớn / số chia 0 / log(0) | LR giảm 10x; thêm epsilon; clip gradient |
| Loss không giảm | LR quá nhỏ hoặc optimizer sai | Tăng LR 10x; thử AdamW trước |
| Loss khởi đầu lớn | Initialization sai | He init cho ReLU, Glorot cho tanh |
| Overfit (train ↓ val ↑) | Model quá to / data ít | Weight decay, dropout, data aug, early stop |
| Saddle point kẹt | Loss đứng yên lâu | Momentum/Adam, đổi LR, biến thiên batch |
| Gradient explosion | Mất mát nổ trong RNN | Clip norm 1.0, gradient 0.5 |
| Gradient vanishing | Mất mát theo thời gian nhỏ dần | LSTM/GRU/Residual/GELU, attention |

---

## 16. Liên hệ chủ đề khác

- **Backprop** — đây chính là chain rule áp dụng cho NN (xem Phase 3 DL sheet 01-pytorch-core)
- **CNN** — convolution = phép tích chập, gradient phức tạp hơn → autograd lo (Phase 3)
- **Transformer** — attention needs softmax + scaled dot product → gradient của softmax cộng với Q/K/V (Phase 3)
- **LLM fine-tune** — AdamW, warmup cosine, gradient clipping là 3 thứ phải mix chuẩn (Phase 4)

---

**Sheet 02 ok · Loa todo CSV.**
