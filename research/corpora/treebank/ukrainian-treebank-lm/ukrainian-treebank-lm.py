# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ukrainian Treebank (Language Modeling) - dataset by Universal Dependencies preprocessed for language modeling"""

import itertools
import operator
import conllu

import datasets


_CITATION = r"""\
@misc{11234/1-3424,
title = {Universal Dependencies 2.7},
author = {Zeman, Daniel and Nivre, Joakim and Abrams, Mitchell and Ackermann, Elia and Aepli, No{\"e}mi and Aghaei, Hamid and Agi{\'c}, {\v Z}eljko and Ahmadi, Amir and Ahrenberg, Lars and Ajede, Chika Kennedy and Aleksandravi{\v c}i{\=u}t{\.e}, Gabriel{\.e} and Alfina, Ika and Antonsen, Lene and Aplonova, Katya and Aquino, Angelina and Aragon, Carolina and Aranzabe, Maria Jesus and Arnard{\'o}ttir, {\t H}{\'o}runn and Arutie, Gashaw and Arwidarasti, Jessica Naraiswari and Asahara, Masayuki and Ateyah, Luma and Atmaca, Furkan and Attia, Mohammed and Atutxa, Aitziber and Augustinus, Liesbeth and Badmaeva, Elena and Balasubramani, Keerthana and Ballesteros, Miguel and Banerjee, Esha and Bank, Sebastian and Barbu Mititelu, Verginica and Basmov, Victoria and Batchelor, Colin and Bauer, John and Bedir, Seyyit Talha and Bengoetxea, Kepa and Berk, G{\"o}zde and Berzak, Yevgeni and Bhat, Irshad Ahmad and Bhat, Riyaz Ahmad and Biagetti, Erica and Bick, Eckhard and Bielinskien{\.e}, Agn{\.e} and Bjarnad{\'o}ttir, Krist{\'{\i}}n and Blokland, Rogier and Bobicev, Victoria and Boizou, Lo{\"{\i}}c and Borges V{\"o}lker, Emanuel and B{\"o}rstell, Carl and Bosco, Cristina and Bouma, Gosse and Bowman, Sam and Boyd, Adriane and Brokait{\.e}, Kristina and Burchardt, Aljoscha and Candito, Marie and Caron, Bernard and Caron, Gauthier and Cavalcanti, Tatiana and Cebiroglu Eryigit, Gulsen and Cecchini, Flavio Massimiliano and Celano, Giuseppe G. A. and Ceplo, Slavomir and Cetin, Savas and Cetinoglu, Ozlem and Chalub, Fabricio and Chi, Ethan and Cho, Yongseok and Choi, Jinho and Chun, Jayeol and Cignarella, Alessandra T. and Cinkova, Silvie and Collomb, Aurelie and Coltekin, Cagr{\i} and Connor, Miriam and Courtin, Marine and Davidson, Elizabeth and de Marneffe, Marie-Catherine and de Paiva, Valeria and Derin, Mehmet Oguz and de Souza, Elvis and Diaz de Ilarraza, Arantza and Dickerson, Carly and Dinakaramani, Arawinda and Dione, Bamba and Dirix, Peter and Dobrovoljc, Kaja and Dozat, Timothy and Droganova, Kira and Dwivedi, Puneet and Eckhoff, Hanne and Eli, Marhaba and Elkahky, Ali and Ephrem, Binyam and Erina, Olga and Erjavec, Tomaz and Etienne, Aline and Evelyn, Wograine and Facundes, Sidney and Farkas, Rich{\'a}rd and Fernanda, Mar{\'{\i}}lia and Fernandez Alcalde, Hector and Foster, Jennifer and Freitas, Cl{\'a}udia and Fujita, Kazunori and Gajdosov{\'a}, Katar{\'{\i}}na and Galbraith, Daniel and Garcia, Marcos and G{\"a}rdenfors, Moa and Garza, Sebastian and Gerardi, Fabr{\'{\i}}cio Ferraz and Gerdes, Kim and Ginter, Filip and Goenaga, Iakes and Gojenola, Koldo and G{\"o}k{\i}rmak, Memduh and Goldberg, Yoav and G{\'o}mez Guinovart, Xavier and Gonz{\'a}lez Saavedra,
Berta and Grici{\=u}t{\.e}, Bernadeta and Grioni, Matias and Grobol, Lo{\"{\i}}c and Gr{\=u}z{\={\i}}tis, Normunds and Guillaume, Bruno and Guillot-Barbance, C{\'e}line and G{\"u}ng{\"o}r, Tunga and Habash, Nizar and Hafsteinsson, Hinrik and Haji{\v c}, Jan and Haji{\v c} jr., Jan and H{\"a}m{\"a}l{\"a}inen, Mika and H{\`a} M{\~y}, Linh and Han, Na-Rae and Hanifmuti, Muhammad Yudistira and Hardwick, Sam and Harris, Kim and Haug, Dag and Heinecke, Johannes and Hellwig, Oliver and Hennig, Felix and Hladk{\'a}, Barbora and Hlav{\'a}{\v c}ov{\'a}, Jaroslava and Hociung, Florinel and Hohle, Petter and Huber, Eva and Hwang, Jena and Ikeda, Takumi and Ingason, Anton Karl and Ion, Radu and Irimia, Elena and Ishola, {\d O}l{\'a}j{\'{\i}}d{\'e} and Jel{\'{\i}}nek, Tom{\'a}{\v s} and Johannsen, Anders and J{\'o}nsd{\'o}ttir, Hildur and J{\o}rgensen, Fredrik and Juutinen, Markus and K, Sarveswaran and Ka{\c s}{\i}kara, H{\"u}ner and Kaasen, Andre and Kabaeva, Nadezhda and Kahane, Sylvain and Kanayama, Hiroshi and Kanerva, Jenna and Katz, Boris and Kayadelen, Tolga and Kenney, Jessica and Kettnerov{\'a}, V{\'a}clava and Kirchner, Jesse and Klementieva, Elena and K{\"o}hn, Arne and K{\"o}ksal, Abdullatif and Kopacewicz, Kamil and Korkiakangas, Timo and Kotsyba, Natalia and Kovalevskait{\.e}, Jolanta and Krek, Simon and Krishnamurthy, Parameswari and Kwak, Sookyoung and Laippala, Veronika and Lam, Lucia and Lambertino, Lorenzo and Lando, Tatiana and Larasati, Septina Dian and Lavrentiev, Alexei and Lee, John and L{\^e} H{\`{\^o}}ng, Phương and Lenci, Alessandro and Lertpradit, Saran and Leung, Herman and Levina, Maria and Li, Cheuk Ying and Li, Josie and Li, Keying and Li, Yuan and Lim, {KyungTae} and Linden, Krister and Ljubesic, Nikola and Loginova, Olga and Luthfi, Andry and Luukko, Mikko and Lyashevskaya, Olga and Lynn, Teresa and Macketanz, Vivien and Makazhanov, Aibek and Mandl, Michael and Manning, Christopher and Manurung, Ruli and Maranduc, Catalina and Marcek, David and Marheinecke, Katrin and Mart{\'{\i}}nez Alonso, H{\'e}ctor and Martins, Andr{\'e} and Masek, Jan and Matsuda, Hiroshi and Matsumoto, Yuji and {McDonald}, Ryan and {McGuinness}, Sarah and Mendonca, Gustavo and Miekka, Niko and Mischenkova, Karina and Misirpashayeva, Margarita and Missil{\"a}, Anna and Mititelu, Catalin and Mitrofan, Maria and Miyao, Yusuke and Mojiri Foroushani, {AmirHossein} and Moloodi, Amirsaeid and Montemagni, Simonetta and More, Amir and Moreno Romero, Laura and Mori, Keiko Sophie and Mori, Shinsuke and Morioka, Tomohiko and Moro, Shigeki and Mortensen, Bjartur and Moskalevskyi, Bohdan and Muischnek, Kadri and Munro, Robert and Murawaki, Yugo and M{\"u}{\"u}risep, Kaili and Nainwani, Pinkey and Nakhl{\'e}, Mariam and Navarro Hor{\~n}iacek, Juan Ignacio and Nedoluzhko,
Anna and Ne{\v s}pore-B{\=e}rzkalne, Gunta and Nguy{\~{\^e}}n Th{\d i}, Lương and Nguy{\~{\^e}}n Th{\d i} Minh, Huy{\`{\^e}}n and Nikaido, Yoshihiro and Nikolaev, Vitaly and Nitisaroj, Rattima and Nourian, Alireza and Nurmi, Hanna and Ojala, Stina and Ojha, Atul Kr. and Ol{\'u}{\`o}kun, Ad{\'e}day{\d o}̀ and Omura, Mai and Onwuegbuzia, Emeka and Osenova, Petya and {\"O}stling, Robert and {\O}vrelid, Lilja and {\"O}zate{\c s}, {\c S}aziye Bet{\"u}l and {\"O}zg{\"u}r, Arzucan and {\"O}zt{\"u}rk Ba{\c s}aran, Balk{\i}z and Partanen, Niko and Pascual, Elena and Passarotti, Marco and Patejuk, Agnieszka and Paulino-Passos, Guilherme and Peljak-{\L}api{\'n}ska, Angelika and Peng, Siyao and Perez, Cenel-Augusto and Perkova, Natalia and Perrier, Guy and Petrov, Slav and Petrova, Daria and Phelan, Jason and Piitulainen, Jussi and Pirinen, Tommi A and Pitler, Emily and Plank, Barbara and Poibeau, Thierry and Ponomareva, Larisa and Popel, Martin and Pretkalnina, Lauma and Pr{\'e}vost, Sophie and Prokopidis, Prokopis and Przepi{\'o}rkowski, Adam and Puolakainen, Tiina and Pyysalo, Sampo and Qi, Peng and R{\"a}{\"a}bis, Andriela and Rademaker, Alexandre and Rama, Taraka and Ramasamy, Loganathan and Ramisch, Carlos and Rashel, Fam and Rasooli, Mohammad Sadegh and Ravishankar, Vinit and Real, Livy and Rebeja, Petru and Reddy, Siva and Rehm, Georg and Riabov, Ivan and Rie{\ss}ler, Michael and Rimkut{\.e}, Erika and Rinaldi, Larissa and Rituma, Laura and Rocha, Luisa and R{\"o}gnvaldsson, Eir{\'{\i}}kur and Romanenko, Mykhailo and Rosa, Rudolf and Roșca, Valentin and Rovati, Davide and Rudina, Olga and Rueter, Jack and R{\'u}narsson, Kristjan and Sadde, Shoval and Safari, Pegah and Sagot, Benoit and Sahala, Aleksi and Saleh, Shadi and Salomoni, Alessio and Samardzi{\'c}, Tanja and Samson, Stephanie and Sanguinetti, Manuela and S{\"a}rg,
Dage and Saul{\={\i}}te, Baiba and Sawanakunanon, Yanin and Scannell, Kevin and Scarlata, Salvatore and Schneider, Nathan and Schuster, Sebastian and Seddah, Djam{\'e} and Seeker, Wolfgang and Seraji, Mojgan and Shen, Mo and Shimada, Atsuko and Shirasu, Hiroyuki and Shohibussirri, Muh and Sichinava, Dmitry and Sigurðsson, Einar Freyr and Silveira, Aline and Silveira, Natalia and Simi, Maria and Simionescu, Radu and Simk{\'o}, Katalin and {\v S}imkov{\'a}, M{\'a}ria and Simov, Kiril and Skachedubova, Maria and Smith, Aaron and Soares-Bastos, Isabela and Spadine, Carolyn and Steingr{\'{\i}}msson, Stein{\t h}{\'o}r and Stella, Antonio and Straka, Milan and Strickland, Emmett and Strnadov{\'a}, Jana and Suhr, Alane and Sulestio, Yogi Lesmana and Sulubacak, Umut and Suzuki, Shingo and Sz{\'a}nt{\'o}, Zsolt and Taji, Dima and Takahashi, Yuta and Tamburini, Fabio and Tan, Mary Ann C. and Tanaka, Takaaki and Tella, Samson and Tellier, Isabelle and Thomas, Guillaume and Torga, Liisi and Toska, Marsida and Trosterud, Trond and Trukhina, Anna and Tsarfaty, Reut and T{\"u}rk, Utku and Tyers, Francis and Uematsu, Sumire and Untilov, Roman and Uresov{\'a}, Zdenka and Uria, Larraitz and Uszkoreit, Hans and Utka, Andrius and Vajjala, Sowmya and van Niekerk, Daniel and van Noord, Gertjan and Varga, Viktor and Villemonte de la Clergerie, Eric and Vincze, Veronika and Wakasa, Aya and Wallenberg, Joel C. and Wallin, Lars and Walsh, Abigail and Wang, Jing Xian and Washington, Jonathan North and Wendt, Maximilan and Widmer, Paul and Williams, Seyi and Wir{\'e}n, Mats and Wittern, Christian and Woldemariam, Tsegay and Wong, Tak-sum and Wr{\'o}blewska, Alina and Yako, Mary and Yamashita, Kayo and Yamazaki, Naoki and Yan, Chunxiao and Yasuoka, Koichi and Yavrumyan, Marat M. and Yu, Zhuoran and Zabokrtsk{\'y}, Zdenek and Zahra, Shorouq and Zeldes, Amir and Zhu, Hanzhi and Zhuravleva, Anna},
url = {http://hdl.handle.net/11234/1-3424},
note = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
copyright = {Licence Universal Dependencies v2.7},
year = {2020} }
"""  # noqa: W605

_DESCRIPTION = """\
Ukrainian part of the Universal Dependencies, specifically preprocessed for the language modeling task. \
The data can be split into documents, paragraphs or sentences. \
Manual selection of the data done by the authors of the dataset makes it suitable for the perplexity evaluation.
Authors of the dataset: Institute for Ukrainian, NGO, org@mova.institute
GitHub: https://github.com/UniversalDependencies/UD_Ukrainian-IU
"""
_HOMEPAGE = "https://github.com/UniversalDependencies/UD_Ukrainian-IU"
_LICENSE = "cc-by-nc-sa-4.0"

_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-IU/master/"
_URLS = {
    "train": _URL + "uk_iu-ud-train.conllu",
    "dev": _URL + "uk_iu-ud-dev.conllu",
    "test": _URL + "uk_iu-ud-test.conllu",
}


class UkrainianTreebankLMConfig(datasets.BuilderConfig):
    """BuilderConfig for UkrainianTreebankLM"""

    def __init__(self, *args, split_by: str = 'paragraph', **kwargs):
        """BuilderConfig for UkrainianTreebankLM.
        Args:
          *args: keyword arguments forwarded to super.
          split_by: one of 'document', 'paragraph', 'sentence'
          **kwargs: keyword arguments forwarded to super.
        """
        super(UkrainianTreebankLMConfig, self).__init__(*args, **kwargs)
        assert split_by in ['document', 'paragraph', 'sentence'], \
            "split_by should be one of 'document', 'paragraph', 'sentence'"

        self.split_by = split_by


class UkrainianTreebankLM(datasets.GeneratorBasedBuilder):
    """Ukrainian Treebank (Language Modeling) - dataset by Universal Dependencies preprocessed for language modeling"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = UkrainianTreebankLMConfig
    BUILDER_CONFIGS = [
        UkrainianTreebankLMConfig(
            name=split_by,
            description=f"Ukrainian Treebank split by {split_by}",
            split_by=split_by,
        )
        for split_by in ['document', 'paragraph', 'sentence']
    ]

    DEFAULT_CONFIG_NAME = "paragraph"

    def _info(self):
        if self.config.split_by == "document":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "document_title": datasets.Value("string")
                }
            )
        elif self.config.split_by == "paragraph":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "document_title": datasets.Value("string"),
                    "paragraph_id": datasets.Value("string")
                }
            )
        elif self.config.split_by == "sentence":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "document_title": datasets.Value("string"),
                    "paragraph_id": datasets.Value("string"),
                    "sentence_id": datasets.Value("string")
                }
            )
        else:
            raise ValueError(f"Invalid split_by value: {self.config.split_by}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        filepaths = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepaths['train'],
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": filepaths['dev'],
                    "split": "dev"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": filepaths['test'],
                    "split": "test"
                },
            ),
        ]

    def _parse(self, filepath):
        with open(filepath, 'r') as f:
            yield from conllu.parse_incr(f, metadata_parsers={"annotation_gap": lambda key, value: (key, value)})

    def _sentence_iterator(self, filepath, split):
        document_id = None
        document_title = None
        paragraph_id = None
        gap_number = 0
        for sentence in self._parse(filepath):
            # TODO how do we handle annotation gaps?
            # if "annotation_gap" in sentence.metadata:
            #     document_id = f"unknown_{gap_number}"
            #     document_title = f"unknown_{gap_number}"
            #     paragraph_id = f"unknown_{gap_number}"
            #     gap_number += 1

            document_id = sentence.metadata.get("newdoc id", document_id)
            document_title = sentence.metadata.get("doc_title", document_title)
            paragraph_id = sentence.metadata.get("newpar id", paragraph_id)

            yield {
                "text": sentence.metadata["text"],
                "document_id": document_id + '_' + split,
                "document_title": document_title,
                "paragraph_id": paragraph_id + '_' + split,
                "sentence_id": sentence.metadata["sent_id"]  # sentence id is always present
            }

    def _generate_examples_sentence(self, sentence_iterator):
        for sentence in sentence_iterator:
            yield sentence["sentence_id"], sentence

    def _generate_examples_paragraph(self, sentence_iterator):
        for key, group in itertools.groupby(sentence_iterator,
                                            operator.itemgetter("paragraph_id")):
            try:
                sentence = next(group)
            except StopIteration:
                continue

            text = sentence["text"] + " ".join([x["text"] for x in group])

            yield key, {
                "text": text,
                "document_id": sentence["document_id"],
                "document_title": sentence["document_title"],
                "paragraph_id": sentence["paragraph_id"],
            }

    def _generate_examples_document(self, sentence_iterator):
        for key, group in itertools.groupby(sentence_iterator,
                                            operator.itemgetter("document_id")):
            try:
                sentence = next(group)
            except StopIteration:
                continue

            text = sentence["text"] + " ".join([x["text"] for x in group])

            yield key, {
                "text": text,
                "document_id": sentence["document_id"],
                "document_title": sentence["document_title"],
            }

    def _generate_examples(self, filepath, split):
        sentence_iterator = self._sentence_iterator(filepath, split)

        if self.config.split_by == "document":
            yield from self._generate_examples_document(sentence_iterator)
        elif self.config.split_by == "paragraph":
            yield from self._generate_examples_paragraph(sentence_iterator)
        elif self.config.split_by == "sentence":
            yield from self._generate_examples_sentence(sentence_iterator)
        else:
            raise ValueError(f"Invalid split_by value: {self.config.split_by}")
