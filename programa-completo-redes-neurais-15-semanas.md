# Programa Completo: Disciplina de Redes Neurais Artificiais
## 15 Semanas | 45 Horas | Mestrado/Doutorado em Computação

---

## **SEMANA 1 — Perceptron e Aprendizado Supervisionado**

### Estrutura da Aula (3 horas)
- **0:00–0:45**: Neurônio de McCulloch & Pitts, Perceptron de Rosenblatt
- **0:45–0:55**: *Intervalo 1*
- **0:55–1:45**: Regra de atualização via gradiente, ativação e convergência
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Linearidade, separabilidade e laboratório
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: História e Formulação (45 min)
- **Contextualização histórica**
  - 1943: McCulloch & Pitts – neurônio binário lógico
  - 1958: Rosenblatt – Perceptron Mark I

- **Modelo matemático**
  - \( y = f(w^T x + b) \)
  - Função de ativação sinal: \( y = \text{sign}(w^T x) \)
  - Interpretação geométrica como hiperplano separador

- **Interpretação geométrica**
  - Comparação com regressão linear
  - Exemplo visual em 2D
  - Noção de hipótese linear separável

#### Bloco 2: Regra de Atualização (50 min)
- **Função de perda e atualização**
  - \( w \leftarrow w + \eta (y - \hat{y})x \)
  - Interpretação como descida de gradiente estocástica

- **Propriedades e Convergência**
  - Teorema de convergência de Rosenblatt
  - Falhas em dados não separáveis

- **Ativação, bias e derivação moderna**
  - Inclusão de bias como entrada constante
  - Transição para perceptrons com funções suaves

#### Bloco 3: Laboratório (55 min)
- **Comparação conceitual**: Perceptron vs. SVM linear
- **Limitações**: XOR e motivação para MLPs
- **Laboratório prático**: Implementação do zero em NumPy

**Leituras**: Aggarwal (2018) Cap. 1, Weidman (2019) Cap. 1

---

## **SEMANA 2 — MLP, Ativações e Backpropagation**

### Estrutura da Aula (3 horas)
- **0:00–0:45**: Redes multicamadas e funções compostas
- **0:45–0:55**: *Intervalo 1*
- **0:55–1:40**: Funções de ativação e suas derivadas
- **1:40–1:50**: *Intervalo 2*
- **1:50–2:50**: Backpropagation completo e laboratório
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Arquitetura MLP (45 min)
- **Limitações do perceptron**: problema XOR
- **Definição de MLP**: \( y = f_L(... f_2(f_1(x))) \)
- **Teorema da universalidade**: aproximação de funções contínuas
- **Interpretando MLP**: engenharia de representação hierárquica

#### Bloco 2: Funções de Ativação (45 min)
- **Sigmoid**: \( \sigma(z) = \frac{1}{1 + e^{-z}} \)
  - Derivada: \( \sigma'(z) = \sigma(z)(1 - \sigma(z)) \)
  - Problemas: saturação, vanishing gradient

- **Tanh**: \( \tanh(z) \), derivada: \( 1 - \tanh^2(z) \)
- **ReLU**: \( \max(0, z) \), derivada: \( \{0, 1\} \)
- **Softmax**: \( \frac{\exp(z_i)}{\sum_j \exp(z_j)} \)

#### Bloco 3: Backpropagation (60 min)
- **Forward pass**: propagação camada a camada
- **Derivada via regra da cadeia**: computational graphs
- **Backpropagation passo a passo**: dedução matemática completa
- **Implementação manual**: NumPy, teste em XOR e multiclasse

**Leituras**: Goodfellow Cap. 6-8, Aggarwal Cap. 3

---

## **SEMANA 3 — Generalização e Regularização**

### Estrutura da Aula (3 horas)
- **0:00–0:50**: Overfitting e bias-variance tradeoff
- **0:50–1:00**: *Intervalo 1*
- **1:00–1:45**: Técnicas de regularização (L1, L2, dropout, early stopping)
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Inicialização, batch normalization e laboratório IMDB
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Overfitting (50 min)
- **Demonstração visual**: curvas de training vs. validation
- **Decomposição Bias-Variance**: \( \text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise} \)
- **Causas**: dados ruidosos, features espúrias, capacidade excessiva

#### Bloco 2: Regularização (45 min)
- **L2 Regularization**: \( L = L_{\text{original}} + \lambda \sum_i w_i^2 \)
- **L1 Regularization**: promove sparsity
- **Dropout**: desativação aleatória com probabilidade \(p\)
- **Early Stopping**: patience, restore_best_weights

#### Bloco 3: Inicialização e Laboratório (55 min)
- **Xavier/Glorot**: \( \text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \)
- **He initialization**: para ReLU
- **Batch Normalization**: \( \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \)
- **Laboratório IMDB**: comparação de técnicas de regularização

**Leituras**: Chollet Cap. 4-5, Aggarwal Cap. 4

---

## **SEMANA 4 — CNNs: Fundamentos e Arquiteturas Clássicas**

### Estrutura da Aula (3 horas)
- **0:00–0:45**: Operações de convolução, padding, pooling, stride
- **0:45–0:55**: *Intervalo 1*
- **0:55–1:45**: LeNet-5, AlexNet e evolução histórica
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Representação hierárquica e laboratório LeNet/MNIST
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Operações Fundamentais (45 min)
- **Motivação**: invariância translacional, redução de parâmetros
- **Convolução 2D**: \( S(i, j) = (K * I)(i, j) = \sum_m \sum_n K(m, n) I(i - m, j - n) \)
- **Stride e padding**: preservação de dimensões
- **Pooling**: max pooling, average pooling

#### Bloco 2: Arquiteturas Clássicas (50 min)
- **LeNet-5 (1998)**: conv → pool → conv → pool → denso
- **AlexNet (2012)**: ReLU, dropout, GPU, data augmentation
- **Análise crítica**: avanços em hardware e datasets

#### Bloco 3: Laboratório (55 min)
- **Hierarquia de features**: bordas → texturas → formas → classes
- **Implementação LeNet**: Keras/PyTorch
- **Visualização**: filtros aprendidos, feature maps

**Leituras**: Chollet Cap. 8, Weidman Cap. 5

---

## **SEMANA 5 — CNNs Modernas: VGG, Inception, ResNet**

### Estrutura da Aula (3 horas)
- **0:00–0:45**: VGG — simplicidade e profundidade
- **0:45–0:55**: *Intervalo 1*
- **0:55–1:45**: InceptionNet — eficiência via paralelismo
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: ResNet — aprendizado residual e laboratório
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Tópico 1: VGG — Simplicidade e Profundidade
- **Princípios**: filtros 3×3 sequenciais
- **Estrutura VGG16/19**: blocos convolucionais repetitivos
- **Contribuições**: modularidade, transfer learning
- **Limitações**: 138M parâmetros, custo computacional

#### Tópico 2: Inception (GoogLeNet) — Paralelismo
- **Módulo Inception**: operações 1×1, 3×3, 5×5, pooling paralelas
- **Convoluções 1×1**: bottleneck, redução de dimensionalidade
- **Estrutura GoogLeNet**: 22 camadas, ~7M parâmetros
- **Global Average Pooling**: substituição de FC layers

#### Tópico 3: ResNet — Aprendizado Residual
- **Degradation problem**: redes profundas têm erro maior
- **Blocos residuais**: \( H(x) = F(x) + x \)
- **Skip connections**: gradientes fluem diretamente
- **Estrutura ResNet-50**: (3, 4, 6, 3) blocos por estágio
- **Laboratório**: comparação VGG vs Inception vs ResNet em CIFAR-10

#### Tópico 4: Técnicas Avançadas
- **Batch Normalization** em CNNs
- **Transfer Learning**: feature extraction vs fine-tuning
- **Data Augmentation**: rotação, flip, crop, mixup

#### Tópico 5: Comparação Quantitativa

| Modelo | Camadas | Parâmetros | Top-5 Error | FLOPs |
|--------|---------|------------|-------------|--------|
| VGG16 | 16 | 138M | 7.3% | 15.5G |
| Inception v3 | 48 | 24M | 5.6% | 5.7G |
| ResNet-50 | 50 | 25M | 5.3% | 4.1G |

**Leituras**: Papers originais VGG, Inception, ResNet

---

## **SEMANA 6 — Redes Recorrentes e Processamento de Sequências**

### Estrutura da Aula (3 horas)
- **0:00–0:45**: Dados sequenciais e arquitetura RNN
- **0:45–0:55**: *Intervalo 1*
- **0:55–1:45**: Vanishing gradient, LSTM, GRU
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Laboratório prático (séries temporais/texto)
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Introdução a RNNs (45 min)
- **Dados sequenciais**: texto, áudio, séries temporais
- **Arquitetura recorrente**: 
  - \( h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h) \)
  - \( y_t = W_{hy} h_t + b_y \)
- **Diagrama unfolded**: timesteps

#### Bloco 2: LSTM e GRU (50 min)
- **Backpropagation Through Time (BPTT)**
- **Vanishing/Exploding gradients**

- **LSTM**: Long Short-Term Memory
  - Gates: input, forget, output
  - Cell state: memória persistente
  - Fórmulas principais

- **GRU**: simplificação do LSTM
- **Bidirecionais e empilhadas**

#### Bloco 3: Laboratório (55 min)
- **Dataset**: previsão de série temporal ou geração de texto
- **Implementação**: RNN → LSTM → GRU
- **Comparação**: performance e convergência
- **Regularização**: dropout em LSTM

**Leituras**: Chollet Cap. 10, Aggarwal Cap. 7

---

## **SEMANA 7 — Seq2Seq e Tradução Automática**

### Estrutura da Aula (3 horas)
- **0:00–0:45**: Arquitetura Encoder-Decoder
- **0:45–0:55**: *Intervalo 1*
- **0:55–1:45**: Teacher Forcing e técnicas de treinamento
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Mecanismos de Atenção (Bahdanau/Luong) e laboratório
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Encoder-Decoder (45 min)
- **Motivação**: tradução, sumarização, chatbots
- **Encoder**: \( \mathbf{c} = q(\mathbf{h}_1, ..., \mathbf{h}_T) \)
- **Decoder**: \( p(y_1, ..., y_T | \mathbf{c}) \)
- **Tokens especiais**: `<BOS>`, `<EOS>`

#### Bloco 2: Teacher Forcing (50 min)
- **Exposure bias**: ground truth vs. predições próprias
- **Teacher forcing**: usar target real durante treinamento
- **Scheduled sampling**: mistura probabilística
- **Professor forcing**: discriminador regulariza diferenças

#### Bloco 3: Atenção e Laboratório (55 min)
- **Problema do bottleneck**: context vector fixo
- **Atenção Bahdanau (additive)**:
  - \( e_{ij} = a(s_{i-1}, h_j) = \mathbf{v}^T \tanh(\mathbf{W}_1 h_j + \mathbf{W}_2 s_{i-1}) \)
  
- **Atenção Luong (multiplicative)**:
  - Dot-product, General, Concat

- **Laboratório**: Seq2Seq com atenção para tradução EN→PT
- **Visualização**: matrizes de atenção (alignment)

**Leituras**: Papers Bahdanau, Luong

---

## **SEMANA 8 — Mecanismos de Atenção Avançados**

### Estrutura da Aula (3 horas)
- **0:00–0:45**: Self-Attention: Query, Key, Value
- **0:45–0:55**: *Intervalo 1*
- **0:55–1:45**: Scaled Dot-Product Attention
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Multi-Head Attention e laboratório
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Fundamentos de Self-Attention (45 min)
- **Diferença de atenção tradicional**: sequência atende a si mesma
- **Analogias**:
  - Sistema de busca (YouTube/Google)
  - Dicionário (HashMap)
  
- **No contexto NLP**:
  - Query (Q): "O que eu quero saber?"
  - Key (K): "Que informação cada palavra oferece?"
  - Value (V): "Qual é o conteúdo real?"

- **Formulação**: \( q_i = x_i W^Q \), \( k_i = x_i W^K \), \( v_i = x_i W^V \)

#### Bloco 2: Scaled Dot-Product (50 min)
- **Cálculo de scores**: \( \text{scores} = QK^T \)
- **Problema de escala**: variância \( \approx d_k \)
- **Solução — Scaled Dot-Product**:
  \[
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  \]

- **Masking**: padding mask, causal mask (look-ahead)
- **Implementação**: NumPy/PyTorch do zero

#### Bloco 3: Multi-Head Attention (55 min)
- **Motivação**: múltiplas "noções de relevância"
- **Processo**:
  1. Projetar em \(h\) conjuntos de Q, K, V
  2. Aplicar attention em cada head
  3. Concatenar outputs
  4. Projeção final \(W^O\)

- **Implementação completa**: classe PyTorch
- **Laboratório**: visualização de attention patterns
- **Análise**: diferentes heads capturam relações diversas

**Leituras**: "Attention Is All You Need" (Vaswani et al., 2017)

---

## **SEMANA 9 — Transformer: Arquitetura Completa**

### Estrutura da Aula (3 horas)
- **0:00–0:50**: Arquitetura Transformer e Positional Encoding
- **0:50–1:00**: *Intervalo 1*
- **1:00–1:45**: Feed-Forward, Residual Connections, Layer Normalization
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Treinamento, Masking e laboratório
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Arquitetura e Positional Encoding (50 min)
- **Paper "Attention Is All You Need" (2017)**: 173k+ citações
- **Estrutura dual**: Encoder (6 camadas) + Decoder (6 camadas)
- **Positional Encoding sinusoidal**:
  \[
  PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
  \]
  \[
  PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
  \]

#### Bloco 2: Feed-Forward e Normalization (45 min)
- **Position-wise FFN**:
  \[
  \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
  \]

- **Residual Connections**: \( \text{Output} = \text{Sublayer}(x) + x \)
- **Layer Normalization**: 
  \[
  \text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
  \]

- **Padrão "Add & Norm"**: \( x = \text{LayerNorm}(x + \text{Sublayer}(x)) \)

#### Bloco 3: Treinamento e Laboratório (55 min)
- **Masking**: padding, look-ahead, cross-attention
- **Teacher forcing** no decoder
- **Label smoothing**, **learning rate schedule** (warmup + decay)

- **Laboratório**: Mini-Transformer para tradução
  - Implementação completa em PyTorch
  - Training loop
  - Visualização de attention weights
  - Comparação com Seq2Seq LSTM

**Leituras**: Paper "Attention Is All You Need" completo

---

## **SEMANA 10 — Large Language Models (LLMs)**

### Estrutura da Aula (3 horas)
- **0:00–0:50**: Evolução dos LLMs: BERT, GPT, T5
- **0:50–1:00**: *Intervalo 1*
- **1:00–1:45**: Scaling Laws e Emergent Abilities
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Prompt Engineering e laboratório
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: BERT, GPT, T5 (50 min)
**BERT (Encoder-only)**:
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Aplicações: classificação, NER, QA

**GPT (Decoder-only)**:
- Causal Language Modeling
- Evolução: GPT-1 (117M) → GPT-4 (~1.7T)
- Zero-shot e few-shot learning

**T5 (Encoder-Decoder)**:
- Text-to-text framework unificado
- Span corruption pré-treinamento

#### Bloco 2: Scaling Laws e Emergent Abilities (45 min)
- **Scaling Laws**: performance previsível
  \[
  L(N) = \left(\frac{N_c}{N}\right)^\alpha
  \]

- **Emergent Abilities**: habilidades imprevisíveis
  - Few-shot learning
  - Aritmética
  - Chain-of-thought reasoning
  - Code generation

- **Debate científico**: mirage vs. real emergence

#### Bloco 3: Prompt Engineering (55 min)
- **Zero-Shot Learning**: instrução sem exemplos
- **Few-Shot Learning**: 1-shot, 3-shot, 5-shot
- **In-Context Learning**: aprender do contexto

- **Componentes de prompt**:
  - System message
  - Context
  - Instruction
  - Examples
  - Input
  - Output indicator

- **Laboratório**: experimentação com API
  - Sentiment analysis
  - Translation
  - Chain-of-thought reasoning
  - System messages

**Leituras**: Papers GPT-3, BERT, "Emergent Abilities of LLMs"

---

## **SEMANA 11 — Autoencoders e VAEs**

### Estrutura da Aula (3 horas)
- **0:00–0:45**: Autoencoders clássicos e latent space
- **0:45–0:55**: *Intervalo 1*
- **0:55–1:45**: VAEs: ELBO, Reparameterization Trick, KL Divergence
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Laboratório: implementando VAE
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Autoencoders Clássicos (45 min)
- **Definição**: rede neural para compressão + reconstrução
- **Componentes**: Encoder \( f_\phi: x \rightarrow z \), Decoder \( g_\theta: z \rightarrow \hat{x} \)
- **Bottleneck**: latent space comprimido
- **Loss**: \( L = \frac{1}{N}\sum_{i=1}^N \|x_i - \hat{x}_i\|^2 \)
- **Latent space**: visualização, propriedades, limitações

#### Bloco 2: VAEs — Teoria (50 min)
- **Paradigma probabilístico**: \( z \sim q_\phi(z|x) = \mathcal{N}(\mu(x), \sigma^2(x)) \)
- **ELBO derivation**:
  \[
  \log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
  \]

- **Loss VAE**:
  \[
  \mathcal{L} = \text{Reconstruction Loss} - \text{KL Divergence}
  \]

- **Reparameterization Trick**:
  \[
  \epsilon \sim \mathcal{N}(0, I), \quad z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon
  \]

#### Bloco 3: Laboratório (55 min)
- **Implementação completa em PyTorch**
- **Dataset**: MNIST
- **Experimentos**:
  - Treinamento e visualização de loss
  - Exploração do latent space 2D
  - Geração de novos dígitos
  - Interpolação no latent space
  - β-VAE: variar peso do KL

**Leituras**: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)

---

## **SEMANA 12 — Generative Adversarial Networks (GANs)**

### Estrutura da Aula (3 horas)
- **0:00–0:50**: Fundamentos de GANs: Minimax Game, Nash Equilibrium
- **0:50–1:00**: *Intervalo 1*
- **1:00–1:45**: DCGAN e técnicas de estabilização
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Mode Collapse e laboratório
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Fundamentos (50 min)
- **Arquitetura dual**: Generator vs. Discriminator
- **Minimax Game**:
  \[
  \min_G \max_D V(D,G) = \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]
  \]

- **Nash Equilibrium**: \( D(x) = 0.5 \) quando \( p_g = p_{data} \)
- **Non-saturating loss**: \( \max_G \log D(G(z)) \)

#### Bloco 2: DCGAN (45 min)
- **Inovações arquiteturais** (Radford et al., 2015):
  1. Strided convolutions (sem pooling)
  2. Batch Normalization (exceto input/output)
  3. Remove FC layers
  4. ReLU (Generator) + LeakyReLU (Discriminator) + Tanh (output)

- **Best practices**:
  - Learning rate: 0.0002
  - Adam com β₁ = 0.5
  - Batch size: 128
  - Weight init: Normal(0, 0.02)

- **Monitoramento**: losses, amostras, IS, FID

#### Bloco 3: Mode Collapse e Laboratório (55 min)
- **Mode Collapse**: Generator produz poucos tipos de outputs
- **Tipos**: total, parcial, rotating
- **Detecção**: visualizar batch, Inception Score
- **Soluções**: minibatch discrimination, Unrolled GAN, WGAN

- **Outros desafios**: vanishing gradients, non-convergence
- **Laboratório**: implementação DCGAN completa
  - Training loop alternado
  - Visualização de evolução
  - Experimentos com hiperparâmetros
  - Interpolação no latent space

**Leituras**: Papers GAN (Goodfellow, 2014), DCGAN (Radford, 2015)

---

## **SEMANA 13 — Diffusion Models e Estado da Arte**

### Estrutura da Aula (3 horas)
- **0:00–0:50**: Forward/Reverse Diffusion e Score Matching
- **0:50–1:00**: *Intervalo 1*
- **1:00–1:45**: DDPM: Training Objective e Implementação
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Latent Diffusion Models (Stable Diffusion) e laboratório
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Fundamentos de Diffusion (50 min)
- **Inspiração**: termodinâmica não-equilibrada
- **Forward process**: adição gradual de ruído
  \[
  q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
  \]

- **Reparametrização direta**:
  \[
  x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
  \]

- **Reverse process**: \( p_\theta(x_{t-1}|x_t) \)
- **Score matching**: conexão teórica

#### Bloco 2: DDPM (45 min)
- **Loss simplificada**:
  \[
  L_{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
  \]

- **Algoritmo de treinamento**: sample timestep aleatório, prever ruído
- **Sampling algorithm**: denoising iterativo de \( x_T \) a \( x_0 \)
- **Arquitetura U-Net**: time embedding, self-attention, ResNet blocks

#### Bloco 3: Latent Diffusion e Stable Diffusion (55 min)
- **Problema**: pixel space é caro
- **Solução**: diffusion no latent space de VAE

- **Componentes Stable Diffusion**:
  1. VAE (Encoder/Decoder)
  2. U-Net (denoising no latent space)
  3. CLIP Text Encoder
  4. Cross-attention conditioning

- **Classifier-Free Guidance**:
  \[
  \tilde{\epsilon} = \epsilon_{\text{uncond}} + s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})
  \]

- **Laboratório**: experimentação com Stable Diffusion
  - Variar guidance scale
  - Negative prompts
  - Número de steps
  - Interpolação entre prompts

**Leituras**: Papers DDPM (Ho, 2020), Stable Diffusion (Rombach, 2022)

---

## **SEMANA 14 — Interpretabilidade, Robustez e Ética**

### Estrutura da Aula (3 horas)
- **0:00–0:50**: Interpretabilidade e XAI (SHAP, LIME, Grad-CAM)
- **0:50–1:00**: *Intervalo 1*
- **1:00–1:45**: Robustez Adversarial: Ataques e Defesas
- **1:45–1:55**: *Intervalo 2*
- **1:55–2:50**: Ética em IA: Viés, Fairness, Privacidade
- **2:50–3:00**: Fechamento

### Conteúdo Detalhado

#### Bloco 1: Interpretabilidade (50 min)
- **Problema da "caixa preta"**
- **Saliency Maps**: \( S = \left|\frac{\partial y_c}{\partial x}\right| \)
- **Grad-CAM**: heatmaps em CNNs
- **LIME**: explicação local via modelo linear
- **SHAP**: Shapley values da teoria dos jogos

#### Bloco 2: Robustez Adversarial (45 min)
- **Exemplos adversariais**: \( x_{adv} = x + \delta \)
- **FGSM**: \( x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J) \)
- **PGD**: FGSM iterativo com projeção
- **C&W**: otimização sofisticada
- **Ataques físicos**: adversarial patches
- **Defesas**:
  - Adversarial training (mais efetiva)
  - Defensive distillation
  - Input transformations
  - Certified defenses

#### Bloco 3: Ética em IA (55 min)
- **Viés algorítmico**:
  - Fontes: dataset, label, model, deployment bias
  - Casos reais: COMPAS, Amazon recruiting, facial recognition

- **Definições de Fairness**:
  - Demographic parity
  - Equal opportunity
  - Equalized odds

- **Privacidade**: differential privacy, federated learning
- **Uso responsável**: deepfakes, environmental impact, automação
- **AI Safety**: alignment, specification gaming

- **Laboratório**: detectando e mitigando viés

**Leituras**: Papers sobre Fairness, Adversarial Examples, AI Ethics

---

## **SEMANA 15 — Apresentações e Perspectivas Futuras**

### Estrutura da Aula (3 horas)
- **0:00–1:00**: Apresentações de Projetos Finais (Parte 1)
- **1:00–1:10**: *Intervalo 1*
- **1:10–2:00**: Apresentações de Projetos Finais (Parte 2)
- **2:00–2:10**: *Intervalo 2*
- **2:10–2:50**: Fronteiras da Pesquisa e Perspectivas Futuras
- **2:50–3:00**: Encerramento

### Conteúdo Detalhado

#### Blocos 1-2: Apresentações de Projetos (110 min total)
- **Formato**: 10 min apresentação + 2-3 min Q&A
- **8-10 projetos** no total

**Categorias de Projetos**:
- Reprodução de paper recente
- Aplicação original
- Estudo comparativo
- Extensão teórica

**Critérios de Avaliação**:
- Implementação técnica (30%)
- Profundidade teórica (25%)
- Qualidade experimental (25%)
- Apresentação (20%)

#### Bloco 3: Fronteiras da Pesquisa (40 min)

**Tópicos Emergentes**:
1. **Modelos Multimodais**: CLIP, GPT-4V, Embodied AI
2. **Efficient AI**: quantization, pruning, distillation
3. **Neuro-Symbolic AI**: integração com raciocínio simbólico
4. **Graph Neural Networks**: dados não-Euclidianos
5. **Continual Learning**: evitar catastrophic forgetting
6. **Foundation Models**: modelos multi-propósito massivos

**Desafios Abertos**:
- Reasoning complexo multi-step
- Sample efficiency
- Interpretabilidade profunda
- Robustez out-of-distribution
- AGI e alignment

**Carreira e Oportunidades**:
- Pesquisa acadêmica vs. indústria
- ML Engineer, MLOps, Data Scientist
- Habilidades valorizadas
- Recursos para continuar aprendendo

#### Encerramento (10 min)
- Reflexão sobre a jornada (Semanas 1-15)
- Princípios para levar adiante
- Feedback da disciplina
- Agradecimentos e despedida

---

## **RESUMO DO PROGRAMA**

### Estrutura Modular
```
Semanas 1-3:  Fundamentos (Perceptron → MLP → Regularização)
Semanas 4-5:  Visão Computacional (CNNs Clássicas e Modernas)
Semanas 6-7:  Sequências (RNNs, LSTMs, Seq2Seq)
Semanas 8-10: Revolução Transformer (Atenção → Transformers → LLMs) ⭐
Semanas 11-13: Modelos Generativos (VAE → GAN → Diffusion) ⭐
Semana 14:    Responsabilidade (Interpretabilidade, Robustez, Ética)
Semana 15:    Integração (Projetos e Perspectivas Futuras)
```

### Avaliação
- **Projeto Final**: 40%
- **Listas de Exercícios**: 30% (3 listas)
- **Paper Review**: 15%
- **Participação**: 15%

### Bibliografia Principal
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning.
3. Aggarwal, C. C. (2018). *Neural Networks and Deep Learning*. Springer.
4. Weidman, S. (2019). *Deep Learning from Scratch*. O'Reilly.

### Papers Fundamentais
- Vaswani et al. (2017): "Attention Is All You Need"
- Goodfellow et al. (2014): "Generative Adversarial Networks"
- Kingma & Welling (2013): "Auto-Encoding Variational Bayes"
- Ho et al. (2020): "Denoising Diffusion Probabilistic Models"
- Devlin et al. (2018): "BERT"
- Brown et al. (2020): "GPT-3"
- Rombach et al. (2022): "Stable Diffusion"

### Pré-requisitos
- Álgebra Linear
- Cálculo Diferencial
- Probabilidade e Estatística
- Python (intermediário)
- Machine Learning (básico)

---

**Elaborado para**: Programa de Pós-Graduação em Computação (Mestrado/Doutorado)  
**Versão**: 2025.2  
**Última atualização**: Outubro 2025
