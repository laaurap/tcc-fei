import unittest

from contexto import eh_contexto_nao_abusivo, neutralizar_expressoes


class ContextoTest(unittest.TestCase):
    def test_expressoes_de_riso_sao_neutralizadas_sem_apagar_o_resto(self):
        casos = [
            "Bla bla você não veio ontem bla bla bla aquela história me mata de rir",
            "Você não veio ontem, mas aquela história me mata de rir.",
            "Aquele meme me mata de rir, vou te mandar",
        ]

        for texto in casos:
            with self.subTest(texto=texto):
                neutralizado, expressoes = neutralizar_expressoes(texto)
                self.assertEqual(expressoes, ["me mata de rir"])
                self.assertIn("rindo muito", neutralizado)
                self.assertNotIn("mata de rir", neutralizado)
                if "não veio ontem" in texto.lower():
                    self.assertIn("você não veio ontem", neutralizado.lower())

    def test_dialogo_respeitoso_e_contexto_nao_abusivo(self):
        casos = [
            "Estou magoada com aquilo e gostaria de conversar para resolvermos.",
            "Fiquei triste com o que aconteceu, mas quero conversar e resolver.",
        ]

        for texto in casos:
            with self.subTest(texto=texto):
                self.assertTrue(
                    eh_contexto_nao_abusivo(
                        score_dialogo=0.95,
                        score_cotidiano=0.10,
                        score_agressao=0.01,
                        texto=texto,
                    )
                )

    def test_conversa_cotidiana_precisa_de_margem_sobre_agressao(self):
        self.assertTrue(
            eh_contexto_nao_abusivo(
                score_dialogo=0.196,
                score_cotidiano=0.448,
                score_agressao=0.356,
                texto="Você não veio ontem, mas aquela história rindo muito.",
            )
        )
        self.assertFalse(
            eh_contexto_nao_abusivo(
                score_dialogo=0.05,
                score_cotidiano=0.45,
                score_agressao=0.40,
                texto="Aquele meme rindo muito, vou te mandar.",
            )
        )

    def test_oferta_de_apoio_e_contexto_nao_abusivo(self):
        self.assertTrue(
            eh_contexto_nao_abusivo(
                score_dialogo=0.553,
                score_cotidiano=0.211,
                score_agressao=0.236,
                texto="Pode contar comigo sempre que precisar.",
            )
        )

    def test_intencao_de_dialogo_nao_esconde_agressao_explicita(self):
        casos = [
            "Fiquei triste com o que aconteceu, mas quero conversar e resolver. Você é inútil.",
            "Fiquei triste com o que aconteceu e quero conversar, mas você nunca pode falar com seus amigos.",
            "Quero conversar e resolver, mas vou te machucar.",
        ]

        for texto in casos:
            with self.subTest(texto=texto):
                self.assertFalse(
                    eh_contexto_nao_abusivo(
                        score_dialogo=0.95,
                        score_cotidiano=0.02,
                        score_agressao=0.03,
                        texto=texto,
                    )
                )

    def test_sinais_de_gaslighting_ou_controle_nao_sao_filtrados(self):
        casos = [
            "Você sempre entende tudo errado.",
            "Talvez o problema seja você.",
            "Você não tem direito de decidir nada sozinha.",
        ]

        for texto in casos:
            with self.subTest(texto=texto):
                self.assertFalse(
                    eh_contexto_nao_abusivo(
                        score_dialogo=0.50,
                        score_cotidiano=0.45,
                        score_agressao=0.05,
                        texto=texto,
                    )
                )

    def test_sinal_contextual_forte_de_agressao_nao_e_filtrado(self):
        self.assertFalse(
            eh_contexto_nao_abusivo(
                score_dialogo=0.80,
                score_cotidiano=0.05,
                score_agressao=0.80,
                texto="Quero conversar e resolver esse desacordo.",
            )
        )

    def test_frase_de_dialogo_com_ataque_implicito_nao_e_filtrada(self):
        self.assertFalse(
            eh_contexto_nao_abusivo(
                score_dialogo=0.54,
                score_cotidiano=0.08,
                score_agressao=0.383,
                texto="Quero conversar e resolver, mas você sempre estraga tudo.",
            )
        )

    def test_ameaca_literal_nao_e_neutralizada_como_expressao_de_riso(self):
        texto = "Vou te matar."
        neutralizado, expressoes = neutralizar_expressoes(texto)
        self.assertEqual(neutralizado, texto)
        self.assertEqual(expressoes, [])

    def test_ameaca_citada_so_e_neutralizada_com_contexto_ficcional_claro(self):
        texto = "No filme, o personagem diz 'vou matar você'; achei a cena engraçada."
        neutralizado, expressoes = neutralizar_expressoes(texto)
        self.assertEqual(neutralizado, "No filme, a cena foi engraçada.")
        self.assertEqual(len(expressoes), 1)

        texto_com_ameaca_adicional = f"{texto} Vou te matar."
        neutralizado, _ = neutralizar_expressoes(texto_com_ameaca_adicional)
        self.assertTrue(neutralizado.endswith("Vou te matar."))

        texto_sem_contexto = "O personagem diz 'vou matar você'."
        neutralizado, expressoes = neutralizar_expressoes(texto_sem_contexto)
        self.assertEqual(neutralizado, texto_sem_contexto)
        self.assertEqual(expressoes, [])


if __name__ == "__main__":
    unittest.main()
