# Tcc
Simulador para aceleradores matriciais. Rodar com "python main.py" ou "python3 main.py".

# Matrix Accelerator Simulator

Este projeto é um **simulador visual interativo** de extensões arquiteturais modernas para aceleração de multiplicação de matrizes, inspirado em arquiteturas como **AMX (Intel)**, **SME (ARM)** e **extensões personalizadas para RISC-V**.

O simulador é focado em ensino e experimentação, oferecendo uma interface gráfica detalhada que mostra cada passo da simulação, registradores internos, matrizes envolvidas e métricas de desempenho.

---

## ✨ Funcionalidades

- **Interface gráfica intuitiva (GUI)** com visualização de matrizes A, B, e C.
- Suporte a múltiplas arquiteturas:
  - ✅ AMX
  - ✅ SME
  - ✅ RISC-V Ext
  - ✅ RISC-V Ext 2
- Simulação passo a passo de:
  - Carga de dados em registradores
  - Execução de produtos externos e escalar
  - Armazenamento do resultado
- Visualização dos registradores vetoriais e acumuladores em cada etapa.
- Métricas de desempenho como quantidade de operações MAC e intensidade computacional.

---

## 📦 Estrutura do Projeto
📁 Projeto
- `main.py`  
  Ponto de entrada da aplicação

- `gui.py`  
  Interface gráfica construída com PyQt5

- `simulator_base.py`  
  Define a base abstrata para simuladores de arquiteturas

- `simulator_factory.py`  
  Fábrica de simuladores baseada no nome da arquitetura

- `amx_simulator.py`  
  Implementação da arquitetura **AMX** (Apple)

- `sme_simulator.py`  
  Implementação da arquitetura **SME** (ARM)

- `riscv_extension_simulator.py`  
  Implementação de uma extensão **RISC-V** com blocos `λ x λ` (lambda)

- `riscv_extension2_simulator.py`  
  Variante da extensão **RISC-V** com registradores `L x 1`, `1 x L` e acumulador `L x L`


## 🧪 Requisitos

- Python 3.8+
- PyQt5
- Numpy

Instale as dependências com:

```bash
pip install pyqt5 numpy
