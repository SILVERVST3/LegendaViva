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

## Como o atraso foi reduzido
Medições no PC com RX 580: cada passada do Whisper levava ~4 s, quase toda no *encoder* (a RX 580 não tem meia precisão no driver Vulkan: `fp16: 0`). A tradução do Google levava 1,2–3 s.

- **Modo rápido da placa (padrão)**: o encoder processa só a parte da janela de 30 s que tem áudio (`audio_ctx`). Nos testes ficou 2,5–4,5× mais rápido. Como isso às vezes faz o modelo "se perder" (pesquisa FUTO whisper-acft e testes próprios), **cada resultado é conferido** (confiança, repetição, tamanho do texto); se parecer ruim, o trecho é refeito com a janela cheia. Se precisar refazer mais de 35% dos trechos, o modo rápido se desliga sozinho.
- **Calibração automática** (só na 1ª vez, ~1 min): mede se o *flash attention* deixa a placa mais rápida ou mais lenta e guarda a melhor opção.
- **Corte assim que a placa fica livre**: em vez de esperar uma pausa longa, o app corta entre palavras quando já juntou áudio equivalente a uma passada. A placa não fica parada.
- **Tradução com reserva**: se o Google não responder em 0,7 s, um segundo endereço do Google é chamado em paralelo e vale a primeira resposta.
- **Legenda provisória (cinza)**: opcional. Desliga sozinha se a placa for lenta demais para ela.

Simulação com os tempos medidos no seu PC: atraso médio fala→legenda caiu de ~9 s para ~5 s.

## Placa de vídeo
- **whisper.cpp + Vulkan** (padrão): usa a placa AMD Radeon. A barra mostra "GPU Vulkan: <placa>". Mantenha o driver **AMD Adrenalin** atualizado.
- **faster-whisper**: processador (ou NVIDIA com CUDA), usado como reserva.

## Ajuste fino
- Se achar que a qualidade caiu, desligue o **Modo rápido da placa** em Configurações e compare.
- No `log.txt`: `[final X s]` = tempo por passada; `modo rápido refeito` = trechos conferidos e refeitos; `calibração` = resultado da calibração.

## Arquivos
Configuração e logs ficam em `%APPDATA%\LegendaViva\` (menu → "Abrir pasta do log"). Se algo der errado, mande o `log.txt` e o `whisper-server.log`.
