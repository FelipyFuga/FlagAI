![FlagAI](logo.png)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6052/badge)](https://bestpractices.coreinfrastructure.org/projects/6052)
[![Python application](https://github.com/FlagAI-Open/FlagAI/actions/workflows/python-app.yml/badge.svg)](https://github.com/FlagAI-Open/FlagAI/actions/workflows/python-app.yml)
![GitHub release (release name instead of tag name)](https://img.shields.io/github/v/release/FlagAI-Open/FlagAI?include_prereleases&style=social)
[简体中文](README.md)

--------------------------------------------------------------------------------

FlagAI (Fast LArge-scale General AI models) é um kit de ferramentas rápido, fácil de usar e extensível para modelos em grande escala. Nosso objetivo é apoiar o treinamento, o ajuste  e a implantação de modelos de grande escala em várias tarefas de downstream com multimodalidade.
 
* Agora ele suporta modelo de representação de imagem de texto [**AltCLIP**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP) e geração de texto para imagem [**AltDiffusion* *](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion) [![Huggingface](https://img.shields.io/badge/🤗-Huggingface%20Space- cyan.svg)](https://huggingface.co/spaces/BAAI/bilingual_stable_diffusion). E suporta **WuDao GLM** com um máximo de 10 bilhões de parâmetros (consulte [Introdução ao GLM](/docs/GLM.md)). Ele também suporta **OPT**, **BERT**, **RoBERTa**, **GPT2**, **T5** e modelos do Huggingface Transformers.

* Ele fornece APIs para baixar e usar rapidamente esses modelos pré-treinados em um determinado texto, ajustá-los em conjuntos de dados amplamente usados ​​coletados de [SuperGLUE](https://super.gluebenchmark.com/) e [CLUE](https ://github.com/CLUEbenchmark/CLUE) e compartilhe-os com a comunidade em nosso hub de modelos. Ele também fornece [aprendizagem rápida](/docs/TUTORIAL_7_PROMPT_LEARNING.md) para algumas tarefas rápidas.   

* Esses modelos podem ser aplicados ao texto (chinês/inglês), para tarefas como classificação de texto, extração de informações, resposta a perguntas, resumo e geração de texto.

* O FlagAI é apoiado pelas três bibliotecas paralelas de dados/modelos mais populares — [PyTorch](https://pytorch.org/)/[Deepspeed](https://www.deepspeed.ai/)/[Megatron-LM]( https://github.com/NVIDIA/Megatron-LM) — com integração perfeita entre eles. Os usuários podem fazer paralelos com seu processo de treinamento/teste com menos de dez linhas de código.

O código é parcialmente baseado em [GLM](https://github.com/THUDM/GLM), [Transformers](https://github.com/huggingface/transformers) e [DeepSpeedExamples](https://github. com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM).

## Novidades
- [10 de novembro de 2022] versão v1.4.0, suporte [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679v1), exemplos em [**AltCLIP**]( https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP) e [**AltDiffusion**](https://github.com/FlagAI-Open/FlagAI/tree/master/examples /AltDiffusion)
- [29 de agosto de 2022] versão v1.3.0, módulo CLIP adicionado e APIs tokenizer redesenhadas em [#81](https://github.com/FlagAI-Open/FlagAI/pull/81)
- [21 de julho de 2022] versão v1.2.0, ViTs são suportados em [#71](https://github.com/FlagAI-Open/FlagAI/pull/71)
- [29 de junho de 2022] versão v1.1.0, suporte para download de OPTs e inferência/ajuste fino [#63](https://github.com/FlagAI-Open/FlagAI/pull/63)
- [17 de maio de 2022] fizemos nossa primeira contribuição em [#1](https://github.com/FlagAI-Open/FlagAI/pull/1)

--------------------------------------------------------------------------------

<!-- toc -->

- [Requisitos e instalação](#requirements-and-installation)
- [Início rápido](#quick-start)
    - [Carregar modelo e tokenizador](#load-model-and-tokenizer)
    - [Previsor](#previsor)
    - [tarefa de geração de texto para imagem](#text-to-image-generation-task)
    - [tarefa NER](#ner-task)
    - [Tarefa de geração de título](#title-generation-task)
    - [tarefa de correspondência semântica](#semantic-matching-task)
- [Modelos e exemplos pré-treinados](#pretrained-models-and-examples)
- [Tutoriais](#tutoriais)
- [Contribuindo](#contribuindo)
- [Fale conosco](#fale-nos)
- [Licença](#licença)

<!-- tocstop -->
## Requerimentos e Instalação
* PyTorch version >= 1.8.0
* Python version >= 3.8
* Para modelos de treinamento/teste em GPUs, você também precisará instalar CUDA e NCCL

Para instalar FlagAI com pip:
```shell
pip install -U flagai
```

- [Opicional]Para instalar o FlagAI e desenvolver localmente:

```shell
git clone https://github.com/FlagAI-Open/FlagAI.git
python setup.py install
```

- [Opicional] Para um treinamento mais rápido, instale o [apex] da NVIDIA (https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- [Opicional] Para otimizadores ZeRO, instale [DEEPSPEED](https://github.com/microsoft/DeepSpeed)
```
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e .
ds_report # check the deespeed status
```
- [Dica] Para ambientes docker de nó único, precisamos configurar portas para o seu ssh. por exemplo, root@127.0.0.1 com porta 7110
```
>>> vim ~/.ssh/config
Host 127.0.0.1
    Hostname 127.0.0.1
    Port 7110
    User root
```
- [Dica] Para ambientes docker de vários nós, gere chaves ssh e copie a chave pública para todos os nós (em `~/.ssh/`)
```
>>> ssh-keygen -t rsa -C "xxx@xxx.com"
```

## Começo rápido
Nós fornecemos muitos modelos que são treinados para executar diferentes tarefas. Você pode carregar esses modelos pelo AutoLoader para fazer previsões. Veja mais em `FlagAI/quickstart`.
## Carregar modelo e tokenizador
Disponibilizamos a classe AutoLoad para carregar o model e o tokenizer rapidamente, por exemplo:
```python
from flagai.auto_model.auto_loader import AutoLoader

auto_loader = AutoLoader(
    task_name="title-generation",
    model_name="BERT-base-en"
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```
Este exemplo é para a tarefa `title_generation`, e você também pode modelar outras tarefas modificando o `task_name`.
Em seguida, você pode usar o modelo e o tokenizer para ajustar ou testar.

## Previsor
Fornecemos a classe `Predictor` para prever diferentes tarefas, por exemplo:

```python
from flagai.model.predictor.predictor import Predictor
predictor = Predictor(model, tokenizer)
test_data = [
   "Quatro minutos após o cartão vermelho, Emerson Royal cabeceou um escanteio para o caminho do desmarcado Kane no segundo poste, que cutucou a bola para seu 12º gol em 17 partidas no derby do norte de Londres. A miséria de Arteta foi agravada dois minutos após o intervalo. momento em que Kane segurou a bola na frente do gol e ajudou Son a acertar um chute além de uma multidão de zagueiros para fazer o 3 a 0. a temporada, e ele parecia perturbado quando foi eliminado a 18 minutos do fim, recebendo palavras de consolo de Pierre-Emile Hojbjerg. Assim que suas frustrações diminuírem, Son e Spurs vão olhar para os dois jogos finais em que precisam apenas de um ponto mais do que o Arsenal para terminar em quarto.",
]

for text in test_data:
    print(
        predictor.predict_generate_beamsearch(text,
                                              out_max_length=50,
                                              beam_size=3))
```

## Modelos e exemplos pré-treinados

* [Text_image_matching with AltCLIP](/examples/AltCLIP/README.md)
* [Geração de texto para imagem com AltDiffusion](/examples/AltDiffusion/README.md)
* [Blank_Filling_QA com GLM](/docs/TUTORIAL_11_GLM_BLANK_FILLING_QA.md)
* [Blank_Filling_QA com GLM](/docs/TUTORIAL_11_GLM_BLANK_FILLING_QA.md)
* [Geração de títulos com GLM](/docs/TUTORIAL_12_GLM_EXAMPLE_TITLE_GENERATION.md)
* [Geração de poesia com GLM-large-ch](docs/TUTORIAL_13_GLM_EXAMPLE_PEOTRY_GENERATION.md)
* [Usando t5-11b e truques de huggingface](docs/TUTORIAL_14_HUGGINGFACE_T5.md)
* [Geração de títulos com RoBerta-WWM](/docs/TUTORIAL_15_BERT_EXAMPLE_TITLE_GENERATION.md)
* [Correspondência Semântica com RoBerta-WWM](/docs/TUTORIAL_16_BERT_EXAMPLE_SEMANTIC_MATCHING.md)
* [NER com RoBerta-WWM](/docs/TUTORIAL_17_BERT_EXAMPLE_NER.md)
* [Escrevendo com GPT-2](/docs/TUTORIAL_18_GPT2_WRITING.md)
* [Geração de título com T5](/docs/TUTORIAL_19_T5_EXAMPLE_TITLE_GENERATION.md)
* [Exemplo de OPT](/examples/opt/README.md)

[//]: # (* [Supported tasks]&#40;/docs/TUTORIAL_20_SUPPORTED_TASKS.md&#41;)


Esta sessão explica como as aulas básicas de NLP funcionam, como você pode carregar modelos pré-treinados para marcar seus
texto, como você pode incorporar seu texto com diferentes incorporações de palavras ou documentos e como você pode treinar seu próprio
modelos de linguagem, modelos de rotulagem de sequência e modelos de classificação de texto. Informe-nos se algo não estiver claro. Veja mais em `FlagAI/examples`.



## Tutoriais
Fornecemos um conjunto de tutoriais rápidos para você começar a usar a biblioteca:
* [Tutorial 1: Como construir e usar o Tokenizer](/docs/TUTORIAL_1_TOKENIZER.md)
* [Tutorial 2: Pipeline de pré-processamento do conjunto de dados](/docs/TUTORIAL_2_DATASET.md)
* [Tutorial 3: Função Principal do Módulo Modelo](/docs/TUTORIAL_3_MODEL.md)
* [Tutorial 4: Personalize o treinador para treinamento paralelo de modelo e dados](/docs/TUTORIAL_4_TRAINER.md)
* [Tutorial 5: Simplifique a inicialização do modelo e do tokenizador usando o Autoloader](/docs/TUTORIAL_5_INSTRUCTIONS_FOR_AutoLoader.md)
* [Tutorial 6: Use algoritmos de inferência disponíveis no mercado com Predictor](/docs/TUTORIAL_6_INSTRUCTIONS_FOR_PREDICTOR.md)
* [Tutorial 7: Use o kit de ferramentas de aprendizado de alerta FlagAI para melhorar o desempenho no SuperGLUE](/docs/TUTORIAL_7_PROMPT_LERANING.md)
* [Tutorial 8: Ambiente de configuração para modelos de treinamento com várias máquinas](/docs/TUTORIAL_8_ENVIRONMENT_SETUP.md)
* [Tutorial 9: Geração de texto com modelos de codificador/decodificador/codificador-decodificador](/docs/TUTORIAL_9_SEQ2SEQ_METHOD.md)
* [Tutorial 10: Como transformar um modelo personalizado em um modelo paralelo estilo megatron-LM](/docs/TUTORIAL_10_MEGATRON.md)

## Contribuindo

Obrigado pelo seu interesse em contribuir! Há muitas maneiras de se envolver;
comece com nossas [diretrizes para contribuidores](CONTRIBUTING.md) e depois
verifique estes [problemas abertos](https://github.com/FlagAI-Open/FlagAI/issues) para tarefas específicas.

## Contact us

<img src="./flagai_wechat.png" width = "200" height = "200"  align=center />

## [License](/LICENSE)
The majority of FlagAI is licensed under the [Apache 2.0 license](LICENSE), however portions of the project are available under separate license terms:

* Megatron-LM is licensed under the [Megatron-LM license](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)
* GLM is licensed under the [MIT license](https://github.com/THUDM/GLM/blob/main/LICENSE)
