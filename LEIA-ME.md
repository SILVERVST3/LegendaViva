# LegendaViva — legendas ao vivo Hebraico → Português

O LegendaViva ouve o som que está tocando no PC (navegador, VLC, qualquer coisa), reconhece a fala em hebraico e mostra a tradução em português numa barra de legenda que fica sempre por cima das outras janelas.

## Como funciona
1. **Captura**: pega o som da saída de áudio do Windows (WASAPI loopback). Não precisa de cabo nem de mixer virtual.
2. **Reconhecimento**: usa o Whisper (na placa de vídeo via whisper.cpp/Vulkan, ou no processador via faster-whisper) com o modelo **ivrit-ai/whisper-large-v3-turbo-ct2**, que foi treinado especialmente para hebraico.
3. **Tradução**: hebraico → português. No modo *Automático* tenta o Google; se ele bloquear, usa o MyMemory; se também falhar, usa o tradutor **offline** NLLB (Meta), que baixa ~600 MB na primeira vez e funciona sem internet. Dá para escolher em Configurações → Tradução.
4. **Legenda**: barra preta semitransparente, que você pode arrastar e redimensionar.

## Gerar o .exe pelo GitHub
1. Crie um repositório novo e envie todos os arquivos desta pasta, **inclusive a pasta `.github`**.
2. O build começa sozinho. Ele também aparece em **Actions → Build LegendaViva (Windows)**.
3. Quando terminar, baixe o artefato **LegendaViva-Windows** (.zip), extraia e rode o `LegendaViva.exe`.

## Primeira execução
- Na primeira vez o modelo é baixado (cerca de 1,6 GB). A barra mostra "Carregando modelo…". Depois disso ele fica guardado no PC.
- O Windows SmartScreen pode avisar "app não reconhecido". Clique em **Mais informações → Executar assim mesmo**.

## Controles da barra
| Ação | Como fazer |
|---|---|
| Mover | arrastar com o botão esquerdo |
| Redimensionar | arrastar o canto ◢ |
| Tamanho da letra | roda do mouse |
| Menu (pausar, mostrar hebraico, configurações, sair) | botão direito |

## Desempenho e placa de vídeo
O app tem dois "motores" de reconhecimento (Configurações → Motor de reconhecimento):

- **whisper.cpp + Vulkan** (padrão quando disponível): usa a placa de vídeo **AMD Radeon (ex.: RX 580)**, Intel ou NVIDIA via Vulkan. Na 1ª vez baixa o modelo de hebraico da ivrit.ai no formato GGML (~1,5 GB). A barra mostra "GPU Vulkan: <nome da placa>" quando está usando a placa.
  - Mantenha o **driver AMD Adrenalin** atualizado (é ele que fornece o Vulkan).
  - **Modo rápido da placa de vídeo**: processa só a parte da janela do Whisper que tem áudio. Nos testes ficou 4–5× mais rápido; pode errar um pouco mais. Vale testar ligado e desligado.
- **faster-whisper**: processador (ou NVIDIA com CUDA). É o reserva automático se o Vulkan falhar.

Dicas:
- O Whisper gasta quase o mesmo tempo para um trecho curto ou longo (exceto no modo rápido). Se o PC não der conta, o app **aumenta o trecho sozinho** (a barra mostra "trechos de Xs (PC lento)"), para não acumular atraso.
- Diminuir o "tamanho do trecho" deixa a legenda mais rápida, mas a tradução fica um pouco pior (frases menores).

## Arquivos
Configuração e logs ficam em `%APPDATA%\LegendaViva\` (menu → "Abrir pasta do log"). Se algo der errado, mande o `log.txt` e o `whisper-server.log`.
