# LegendaViva — legendas ao vivo Hebraico → Português

O LegendaViva ouve o som que está tocando no PC (navegador, VLC, qualquer coisa), reconhece a fala em hebraico e mostra a tradução em português numa barra de legenda que fica sempre por cima das outras janelas.

## Como funciona
1. **Captura**: pega o som da saída de áudio do Windows (WASAPI loopback). Não precisa de cabo nem de mixer virtual.
2. **Reconhecimento**: usa o Whisper (faster-whisper) com o modelo **ivrit-ai/whisper-large-v3-turbo-ct2**, que foi treinado especialmente para hebraico.
3. **Tradução**: hebraico → português pelo Google Tradutor (precisa de internet).
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

## Desempenho (importante)
- **Com placa NVIDIA**: as legendas aparecem cerca de 1 a 2 s depois do fim de cada frase, mais o tamanho do trecho (5 s por padrão). Para usar a GPU, instale o **CUDA 12 + cuDNN 9** da NVIDIA. Sem eles, o app usa o processador sozinho.
- **Só processador**: o modelo turbo funciona em CPUs boas (i5/Ryzen 5 recentes ou melhores). Se a barra mostrar "atrasado", vá em Configurações e escolha `medium` ou `small` (mais rápidos, mas erram mais em hebraico).
- Diminuir o "tamanho do trecho" deixa a legenda mais rápida, mas a tradução fica um pouco pior porque recebe frases menores.

## Arquivos
Configuração e log ficam em `%APPDATA%\LegendaViva\` (menu → "Abrir pasta do log"). Se algo der errado, mande o `log.txt`.
