"""
gerar_teste_congelado.py
Gera dataset/teste_congelado.csv: frases NOVAS, escritas separadamente do
dataset de treino, usadas SÓ para avaliar. NUNCA copie estas frases para
gerar_dataset_proprio.py. O treinar_combinado.py descarta do treino qualquer
frase que apareça aqui, como proteção.

Colunas: texto, rotulo (1 = violência psicológica, 0 = não), categoria
"""

import csv

TESTE = [
    # --- Positivos: ameaça / perseguição ---
    ("Se você terminar comigo eu vou atrás de você em qualquer lugar que for.", 1, "ameaça"),
    ("Pode me bloquear, eu dou um jeito de te encontrar.", 1, "ameaça"),
    ("Não adianta se esconder, eu vou te achar e você vai ver.", 1, "ameaça"),
    ("Mais uma mentira dessas e você não sai inteira daqui.", 1, "ameaça"),
    ("Vou ficar na frente da sua casa todo dia até você voltar.", 1, "ameaça"),
    ("Se você me trocar, vai pagar por isso, pode ter certeza.", 1, "ameaça"),
    ("Eu sei o seu horário, vou te esperar na saída do trabalho.", 1, "ameaça"),
    ("Se contar pra polícia, a sua família vai sofrer.", 1, "ameaça"),
    # --- Positivos: controle ---
    ("Antes de sair de casa você precisa me pedir autorização.", 1, "controle"),
    ("Deleta o Instagram, não quero você falando com ninguém.", 1, "controle"),
    ("Você vai me mandar foto de onde está de hora em hora.", 1, "controle"),
    ("Não vou deixar você usar essa saia, troca de roupa agora.", 1, "controle"),
    ("Fica sem celular, eu guardo pra você não conversar com seus amigos.", 1, "controle"),
    ("A partir de hoje eu cuido do seu dinheiro, você não mexe mais.", 1, "controle"),
    # --- Positivos: gaslighting ---
    ("Isso não aconteceu desse jeito, você está distorcendo tudo de novo.", 1, "gaslighting"),
    ("Você está ficando paranoica, ninguém falou nada disso.", 1, "gaslighting"),
    ("Eu jamais disse isso, você inventou só pra brigar comigo.", 1, "gaslighting"),
    ("Sua memória é péssima, tudo que você conta é mentira.", 1, "gaslighting"),
    # --- Positivos: xingamento / humilhação ---
    ("Você é uma completa imprestável, só serve pra dar trabalho.", 1, "xingamento"),
    ("Cala a boca, seu inútil, ninguém pediu a sua opinião.", 1, "xingamento"),
    ("Que mulher burra, não consegue entender uma coisa simples.", 1, "xingamento"),
    ("Você é uma vergonha, ninguém quer ser visto do seu lado.", 1, "humilhação"),
    ("Você não tem capacidade de decidir nada, deixa que eu resolvo tudo.", 1, "humilhação"),
    ("Sem mim você não é ninguém, nunca esqueça disso.", 1, "humilhação"),
    ("Eu tenho vergonha de dizer que sou casado com você.", 1, "humilhação"),
    # --- Positivos: chantagem emocional ---
    ("Se você for na viagem com as amigas, eu vou me machucar e você vai ser a culpada.", 1, "chantagem_emocional"),
    ("Depois de tudo que eu passei por você, você me larga assim?", 1, "chantagem_emocional"),
    ("Se me deixar eu não respondo por mim, e vai ser tudo culpa sua.", 1, "chantagem_emocional"),
    ("Ninguém vai te amar como eu, volta ou vai ficar sozinha pra sempre.", 1, "chantagem_emocional"),
    # --- Positivos: invalidação ---
    ("Pare de chorar por besteira, ninguém aguenta esse seu drama.", 1, "invalidação"),
    ("Você exagera em tudo, isso nem é motivo pra ficar triste.", 1, "invalidação"),
    ("Nossa, de novo se fazendo de vítima? Cresce.", 1, "invalidação"),
    ("Esse seu sofrimento é puro show pra chamar atenção.", 1, "invalidação"),
    # --- Negativos: hipérbole / brincadeira ---
    ("Vou te seguir até a padaria pra ganhar um pedaço do seu pão, kkk.", 0, "hiperbole"),
    ("Se não me mandar a foto do show agora eu fico te cobrando até domingo!", 0, "hiperbole"),
    ("Juro que se você contar o final da série eu nunca mais te perdoo, brincadeira.", 0, "hiperbole"),
    ("Vou te sequestrar pra ir comigo na praia no feriado.", 0, "hiperbole"),
    ("Se você não vier no meu aniversário eu vou na sua casa te buscar de pijama.", 0, "hiperbole"),
    ("Esse chocolate tá me matando, eu não consigo parar de comer!", 0, "hiperbole"),
    ("Vou te encher de beijo quando eu te ver, prepara.", 0, "hiperbole"),
    ("Manda a foto da sua nova cachorrinha ou eu vou aí buscar ela!", 0, "hiperbole"),
    ("Morri de rir com o vídeo que você mandou.", 0, "idiomatica"),
    ("Esse sol de meio-dia vai me matar, preciso de água.", 0, "idiomatica"),
    ("Tô morrendo de sono, vou deitar cedo hoje.", 0, "idiomatica"),
    ("Essa prova me deixou morta de nervoso.", 0, "idiomatica"),
    # --- Negativos: ficção / notícia / citação ---
    ("A personagem do livro é perseguida pelo ex, a autora retrata bem o medo dela.", 0, "ficcao"),
    ("Assisti a um documentário sobre relacionamentos abusivos, super esclarecedor.", 0, "ficcao"),
    ("Na peça, o ator grita 'eu vou te destruir', foi muito intenso.", 0, "ficcao"),
    ("Li uma matéria dizendo que o controle do celular do parceiro é um sinal de abuso.", 0, "ficcao"),
    # --- Negativos: desabafo / diálogo saudável ---
    ("Fiquei chateada quando você esqueceu da data, podemos conversar sobre isso?", 0, "dialogo"),
    ("Desculpa ter gritado, eu perdi a paciência e não devia.", 0, "dialogo"),
    ("Acho que estou precisando de um tempo sozinho, mas não quero te magoar.", 0, "dialogo"),
    ("Eu me senti pouco ouvido na conversa de ontem, queria te explicar.", 0, "dialogo"),
    ("Não concordo com a sua decisão, mas a escolha é sua e eu respeito.", 0, "dialogo"),
    ("Estou com ciúme, mas sei que é um problema meu, vou trabalhar nisso.", 0, "dialogo"),
    # --- Negativos: cotidiano ---
    ("Chego em casa por volta das sete, quer que eu leve alguma coisa?", 0, "cotidiano"),
    ("Você pode me ajudar com o trabalho de história amanhã?", 0, "cotidiano"),
    ("Reservei a mesa do restaurante para as oito horas.", 0, "cotidiano"),
    ("Avisa quando chegar, fica tranquila e dirige com cuidado.", 0, "cotidiano"),
    ("A feira de ciências foi ótima, meu grupo ficou em segundo lugar.", 0, "cotidiano"),
    ("Posso te ligar depois do almoço?", 0, "cotidiano"),
    ("Obrigada pela ajuda de hoje, você salvou o meu dia.", 0, "cotidiano"),
    ("O ônibus atrasou, vou chegar uns dez minutos depois.", 0, "cotidiano"),
    ("Vamos assistir aquele filme que você indicou no sábado?", 0, "cotidiano"),
    ("Hoje vou ficar estudando, depois te conto como foi.", 0, "cotidiano"),
    # --- Negativos: discordância / limite saudável ---
    ("Não vou poder sair hoje, estou muito cansada, fica pra amanhã.", 0, "limite"),
    ("Prefiro que você não fale assim comigo, isso me incomoda.", 0, "limite"),
    ("Esse trabalho precisa de ajustes, vou te apontar o que melhorar.", 0, "limite"),
    ("Discordo do seu plano, acho que vai custar caro demais.", 0, "limite"),
]


def main():
    caminho = "dataset/teste_congelado.csv"
    with open(caminho, "w", encoding="utf-8-sig", newline="") as arq:
        w = csv.writer(arq)
        w.writerow(["texto", "rotulo", "categoria"])
        w.writerows(TESTE)
    pos = sum(1 for _, r, _ in TESTE if r == 1)
    print(f"{len(TESTE)} frases ({pos} positivas, {len(TESTE) - pos} negativas) -> {caminho}")


if __name__ == "__main__":
    main()
