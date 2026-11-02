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
- **Legenda provisória (cinza)**: enquanto a pessoa fala, o app reconhece de novo o áudio da frase toda vez que a placa fica livre, e mostra a tradução provisória em cinza. Quando a frase termina, faz uma passada final mais caprichada e o texto fica **branco**. Técnica baseada no Whisper-Streaming (Universidade Carlos, Praga).
- O ritmo das passadas se ajusta sozinho à velocidade da placa: nunca acumula atraso.
- Frases cortadas nas pausas da fala (300 ms), para não partir palavras.
- Tradução com conexão reaproveitada (sem refazer a conexão segura a cada frase) e cache: se a frase final é igual à última provisória, a tradução é instantânea.
- Passadas provisórias usam decodificação gulosa e sem re-tentativas; a final usa beam search quando a placa dá conta.

## Placa de vídeo
- **whisper.cpp + Vulkan** (padrão): usa a placa AMD Radeon (ex.: RX 580). A barra mostra "GPU Vulkan: <placa>". Mantenha o driver **AMD Adrenalin** atualizado.
- **faster-whisper**: processador (ou NVIDIA com CUDA), usado como reserva.
- "Encoder encurtado" ficou como **experimental e desligado**: pesquisas mostram que com o modelo turbo ele multiplica os erros.

## Ajuste fino
- **Tamanho máx. da frase**: com legenda provisória, 5–8 s é o ideal (frases maiores = tradução melhor, sem aumentar o atraso).
- No `log.txt`, as linhas `[parcial X s]` e `[final X s]` mostram quanto a placa leva por passada.

## Arquivos
Configuração e logs ficam em `%APPDATA%\LegendaViva\` (menu → "Abrir pasta do log"). Se algo der errado, mande o `log.txt` e o `whisper-server.log`.
