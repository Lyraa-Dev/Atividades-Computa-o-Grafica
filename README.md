Simulação em Python para responder a ficha 01 da cadeira de Computação Grafica.

Grupo: José Ailton, Eduardo Silva e Ricardo Lyra 

Esta lista de respostas está escrita da seguinte forma R + Numero da questão + letra da questão_ficha1.py.
Exemplo R1A_ficha1.py se refere a resposta da primeira questão letra A da ficha 1

As perguntas referentes a matrizes em coordenadas homogêneas do operador afim e etc estarão sendo respondidas no terminal ao rodar o programa, infelizmente não me atentei em algumas questões onde o senhor pediu para simular todas ao mesmo tempo e acabei fazendo isso separadamente XDD, tentei identificar em cada parte do codigo o que cada função fazia ou do que se tratava cada parte do codigo 

Como rodar
- Certifique-se de ter Python 3.9+ instalado.
- Instale as dependências:
pip install numpy matplotlib
- Salve o código Python em um arquivo
- Execute:
python R1A_ficha1.py (por exemplo)
- A janela do Matplotlib abrirá mostrando a simulação e as respostas serão exibidas no terminal:

Dificuldades encontradas
- Definição das matrizes em Python:
Criar matrizes homogêneas em python foi mais complexo do que imaginava, precisando verificar fórmulas que compõem cada tipo de modificação na matriz.

- Reflexão em relação a um plano:
Na internet consegui identificar diversas formas de criar um codigo para calcular a normais de planos para que possamos realizar as transformações, porém aprender como funciona (sem mencionar os calculos matematicos) levou um pouco de tempo

- Simulação visual:
Representar algumas das formas realizando as transformações solicitadas foi um pouco complicado, onde foi necessario criar alguns auxilios visuais como trajetoria marcação de pontos e visualização de planos, tentando tambem suavizar ao maximo a simulação para que fosse possivel compreender o que estava sendo realizado (Sem mencionar a 4ª questão que nos levou mais tempo que o esperado para criar a simulação que ao menos parecesse ser correta)

Questão 4 -B

Com essa ideia o plano A e B estariam rotacionando em torno do eixo D sendo o eixo A no sentido horario e B no sentido anti-horario com a mesma velocidade angular ω, com isso precisariamos fazer com que a definicação dos planos A e B se movimentassem de forma continua as normais e pontos de referencia do plano, na simulação isso faria com que os planos estivessem se "torcendo" junto com a cobrinha e alterando o ponto de colisão da fenda constantemente

