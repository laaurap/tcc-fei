"""
knowledge_base.py
Base de referência sobre violência psicológica, usada pelo RAG.

Os textos abaixo são resumos escritos com base em fontes acadêmicas e
legais reais (citadas em cada verbete no campo "fonte"). Não são citações
literais — são explicações próprias, reescritas para uso técnico no
sistema. Para a monografia, cite as fontes originais listadas abaixo,
não este arquivo.

Fontes consultadas:
  - BRASIL. Lei nº 11.340, de 7 de agosto de 2006 (Lei Maria da Penha), Art. 7º, II.
  - BRASIL. Código Penal, Art. 147-B (incluído pela Lei nº 14.188/2021).
  - HIRIGOYEN, M-F. (2006). Apud artigo "Violência Psicológica no Contexto
    Familiar e a Lei Maria da Penha", Revista FT, ISSN 1678-0817.
  - CHAGAS, A. P.; MARTINS, M. das G. T. (2022). Fenômeno Gaslight: da
    manipulação psicológica ao empoderamento feminino. Revista
    Ibero-Americana de Humanidades, Ciências e Educação, 8(3), 579-596.
    https://doi.org/10.51891/rease.v8i3.4617
  - SARKIS, S. M. (2019). O fenômeno Gaslighting. 1. ed. Cultrix.
  - MELO, I.; ORNELAS. A violência psicológica e a Lei Maria da Penha.
    Conjur, 2021. https://www.conjur.com.br/2021-ago-05/igor-melo-violencia-psicologica-lei-maria-penha/

IMPORTANTE: este é um ponto de partida. Recomenda-se que o orientador
valide/complemente com fontes acadêmicas adicionais específicas do curso.
"""

REFERENCIAS = [
    {
        "id": "ref_definicao_geral",
        "categoria": "geral",
        "titulo": "Definição legal de violência psicológica",
        "texto": (
            "O artigo 147-B do Código Penal, incluído pela Lei 14.188/2021, "
            "tipifica a violência psicológica como causar dano emocional que "
            "prejudique o desenvolvimento da vítima, ou que vise degradar ou "
            "controlar suas ações, comportamentos, crenças e decisões, por "
            "meio de ameaça, constrangimento, humilhação, manipulação, "
            "isolamento, chantagem, ridicularização ou limitação do direito "
            "de ir e vir. Essa tipificação complementa a definição já "
            "prevista no Art. 7º, II, da Lei Maria da Penha (Lei "
            "11.340/2006)."
        ),
        "fonte": "BRASIL, Código Penal, Art. 147-B; Lei 11.340/2006, Art. 7º, II",
    },
    {
        "id": "ref_hirigoyen",
        "categoria": "geral",
        "titulo": "Conceito de violência psicológica (Hirigoyen)",
        "texto": (
            "A psiquiatra Marie-France Hirigoyen caracteriza a violência "
            "psicológica como um padrão de atitudes e expressões que buscam "
            "rebaixar ou negar a identidade e o valor de outra pessoa. Esse "
            "padrão costuma ser sutil e repetitivo, o que dificulta seu "
            "reconhecimento pela própria vítima em estágios iniciais."
        ),
        "fonte": "HIRIGOYEN, M-F. (2006), apud Revista FT, ISSN 1678-0817",
    },
    {
        "id": "ref_humilhacao_ameaca",
        "categoria": "humilhação",
        "titulo": "Humilhação, ameaça e constrangimento",
        "texto": (
            "A legislação brasileira reconhece a humilhação, o "
            "constrangimento e a ameaça como condutas que configuram "
            "violência psicológica quando usadas para controlar ou degradar "
            "a vítima. Frases que ridicularizam a capacidade ou o caráter da "
            "pessoa, de forma recorrente, são indicadores dessa conduta."
        ),
        "fonte": "BRASIL, Código Penal, Art. 147-B",
    },
    {
        "id": "ref_ameaca",
        "categoria": "ameaça",
        "titulo": "Ameaça como violência psicológica",
        "texto": (
            "Ameaças de dano físico ou de consequências graves à vítima ou "
            "a terceiros são expressamente citadas na legislação como uma "
            "das formas de violência psicológica, por gerarem medo "
            "constante e comprometerem a autodeterminação da pessoa "
            "ameaçada."
        ),
        "fonte": "BRASIL, Código Penal, Art. 147-B; Lei 11.340/2006, Art. 7º, II",
    },
    {
        "id": "ref_gaslighting_1",
        "categoria": "gaslighting",
        "titulo": "Gaslighting como manipulação da realidade",
        "texto": (
            "Chagas e Martins (2022) descrevem o gaslighting como uma forma "
            "de manipulação psicológica sistemática, na qual o agressor "
            "distorce fatos e nega acontecimentos para fazer a vítima "
            "duvidar da própria percepção e memória. Os autores destacam "
            "que a prática compromete a autoestima da vítima, sendo o "
            "sentimento de vergonha um efeito predominante identificado na "
            "literatura revisada."
        ),
        "fonte": "CHAGAS; MARTINS (2022), DOI: 10.51891/rease.v8i3.4617",
    },
    {
        "id": "ref_gaslighting_2",
        "categoria": "gaslighting",
        "titulo": "Gaslighting como estratégia de controle contínuo",
        "texto": (
            "Sarkis (2019) descreve o gaslighting como uma estratégia usada "
            "por pessoas manipuladoras para distorcer a verdade de forma "
            "contínua, com o efeito de manter a vítima sob controle "
            "psicológico, minando gradualmente sua confiança no próprio "
            "julgamento."
        ),
        "fonte": "SARKIS, S. M. (2019). O fenômeno Gaslighting. Cultrix.",
    },
    {
        "id": "ref_controle_isolamento",
        "categoria": "controle",
        "titulo": "Controle e isolamento como violência psicológica",
        "texto": (
            "A legislação penal brasileira inclui o isolamento e a "
            "limitação do direito de ir e vir entre as condutas que "
            "caracterizam violência psicológica, quando usadas para "
            "controlar as ações e decisões da vítima. O controle excessivo "
            "sobre com quem a pessoa fala ou para onde vai é considerado um "
            "mecanismo de subjugação."
        ),
        "fonte": "BRASIL, Código Penal, Art. 147-B",
    },
    {
        "id": "ref_chantagem_manipulacao",
        "categoria": "chantagem_emocional",
        "titulo": "Chantagem e manipulação emocional",
        "texto": (
            "A chantagem é citada expressamente na tipificação penal como "
            "uma das condutas que configuram violência psicológica, ao lado "
            "da manipulação. Ocorre quando o agressor usa culpa, medo ou "
            "ameaças veladas para controlar o comportamento da vítima."
        ),
        "fonte": "BRASIL, Código Penal, Art. 147-B",
    },
    {
        "id": "ref_invalidacao_ridicularizacao",
        "categoria": "invalidação",
        "titulo": "Ridicularização e invalidação emocional",
        "texto": (
            "A ridicularização dos sentimentos, opiniões ou comportamentos "
            "da vítima é reconhecida na legislação como conduta que causa "
            "dano emocional. Tratar reações legítimas como exagero ou "
            "drama, de forma recorrente, impede a vítima de reconhecer e "
            "processar a própria situação de abuso."
        ),
        "fonte": "BRASIL, Código Penal, Art. 147-B",
    },
    {
        "id": "ref_xingamento",
        "categoria": "xingamento",
        "titulo": "Agressão verbal e constrangimento",
        "texto": (
            "O constrangimento por meio de insultos e palavras ofensivas "
            "dirigidas à vítima, quando repetido, integra o padrão de "
            "condutas que a legislação brasileira caracteriza como "
            "violência psicológica, distinguindo-se de um desentendimento "
            "pontual pela sua sistematicidade."
        ),
        "fonte": "BRASIL, Código Penal, Art. 147-B",
    },
]