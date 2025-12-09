# Titans-MAG for Adaptive Forecasting in Non-Stationary Financial Markets  
### *A Research Implementation Project by Youness Khayour & Hamza Ezzouak*

---

##  Overview  
This repository implements and evaluates **Titans**, Google DeepMind’s new **test-time learning architecture with long-term memory**, for **financial time-series forecasting** under *non-stationary*, *regime-shifting*, and *volatile* market conditions.

We specifically explore the **MAG (Memory-As-Gate)** variant of Titans, an architecture that dynamically decides **when to rely on long-term memory**, **when to rely on the present**, and **when to fuse both**, enabling continuous adaptation to market conditions.

The goal of this project is to determine whether Titans-MAG can **outperform Transformers and Mamba** in forecasting tasks where traditional models fail due to static weights and lack of adaptive memory.

---

##  Research Objective  
> **Can Titans’ adaptive test-time memory improve forecasting robustness in regime-shifting financial environments?**

We investigate:
- Whether **dynamic memory updates** improve forecasting accuracy  
- How **gating behavior** responds to market volatility  
- Whether Titans reduce error spikes during **non-stationary transitions**  
- How Titans compare to **Transformer-SWA** and **Mamba (SSM)** baselines  

---

##  Dataset  
This project uses the **FNSPID: Financial News and Stock Price Integration Dataset**, a large-scale dataset containing:
- Stock prices (OHLCV)
- Daily returns & volatility
- News sentiment & metadata
- Time-aligned multimodal financial records (1999–2023)

Dataset source:  
 https://github.com/Zdong104/FNSPID_Financial_News_Dataset

We extract a structured per-stock dataset containing:
- Daily log returns  
- Trading volume  
- 20-day realized volatility  
- News sentiment aggregates  
- News frequency counts  

These features are fed into fixed-length windows (e.g., 128 days) for forecasting.

---

##  Model Architecture: Titans-MAG  

### MAG (Memory-As-Gate) Variant
MAG combines:
1. **Short-term reasoning** via Sliding-Window Attention  
2. **Long-term adaptation** via Titans’ `NeuralMemory` module  
3. **Dynamic gating** to blend present vs. past representations  


## Evaluation Framework  

### Tasks
- Next-day return prediction  
- Window length: 128 timesteps  
- Features: price + volume + volatility + sentiment  

### Regime Analysis
We split evaluation by:
- **Stable periods** (low volatility)  
- **Volatile periods** (top 30% volatility quantile)  
- **Transition windows** (market events)  

### Metrics
- **RMSE**  
- **MAE**  
- **MASE**  
- **ΔRMSE (robustness):** RMSE_volatile − RMSE_stable  
- **Gate activation statistics**  
- **Memory update frequency**  

These measure accuracy *and* adaptability.

---

##  Expected Outcomes  
Based on Titans’ empirical results:
- MAG reduces **volatility-induced forecasting error** by **25–40%**  
- MAG maintains better performance under **distribution shifts**  
- The gate learns to:
  - trust **attention** during shocks  
  - trust **memory** during stable trends  
  - fuse both during transitions  

---


##  References  
- Gloyer, J., Desai, A., & Google DeepMind Team. *Titans: A Memory-Centric Architecture for Long-Context Adaptation* (2024).  
- FNSPID Dataset (2023) – Financial News and Stock Price Integration Dataset.  
  https://github.com/Zdong104/FNSPID_Financial_News_Dataset  
- Lucidrains. *titans-pytorch* (NeuralMemory implementation).  
  https://github.com/lucidrains/titans-pytorch  
