# DifTransNet: A Differential Cross-Scale Transformer Network for Infrared Small Target Detection

🎉 Our paper **"DifTransNet: A Differential Cross-Scale Transformer Network for Infrared Small Target Detection"** has been accepted.

This repository provides the official implementation of DifTransNet, a transformer-based framework designed for infrared small target detection. The proposed method focuses on enhancing target–background discrimination by introducing differential cross-scale interaction and adaptive feature fusion strategies.

## Overview

Infrared small target detection suffers from weak target signals, low signal-to-noise ratios, and complex background interference. To address these challenges, DifTransNet introduces two key components:

- **Differential Cross Transformer Block (DCTB):**
  A cross-scale transformer module that constructs dual-query branches to separately model target-related responses and background-related responses. Through differential aggregation, DCTB enhances discriminative target features while suppressing background clutter.

- **Directional Multi-Scale Fusion Module (DMSF):**
  A feature fusion module that combines directional spatial modeling and adaptive channel–spatial weighting to preserve weak target structures during multi-level feature reconstruction.

The overall architecture and detailed module designs are illustrated below.

## Network Architecture

![DifTransNet Architecture](./figures/framework.png)

## Differential Cross Transformer Block (DCTB)

DCTB performs multi-scale feature interaction through a shared key-value representation and dual-query attention mechanism. One branch emphasizes local target-sensitive variations, while the other captures background contextual information. The final representation is obtained through adaptive differential aggregation, enabling explicit target enhancement and background suppression.

![DCTB](./figures/DCTB.png)

## Directional Multi-Scale Fusion Module (DMSF)

DMSF is designed to improve decoder feature reconstruction. By jointly modeling directional spatial dependencies and channel-wise importance, it effectively integrates shallow spatial details and deep semantic information, which is beneficial for recovering extremely small infrared targets.

![DMSF](./figures/DMSF.png)
