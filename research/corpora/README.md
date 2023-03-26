# Available Ukrainian Language Corpora

## All outgoing links from Wikipedia

We would rather need a dump to run such an experiment. After the conversation with Krzysztof I learned, that there might be not that many links, they might lead to too specific places, and they might not be of any use.

**Pros:**

* similar to WebText approach

**Cons:**

* may not be that many links -> not that much data
* linked pages may not be diverse enough
* needs a lot of preprocessing

**Status: frozen** 

## Ukraїner corpus

**Pros:**

* quality data written by humans
* a lot of punctuation
* conversational style

**Cons:**

* a lot of surzhyk
* lots of out of vocabulary words

**Status: discussion about the data access**

## Subtitles

Found [this](https://www.sketchengine.eu/corpora-and-languages/ukrainian-text-corpora/) so far. Subtitles part is really small, but there are links to corpora consisting of 2.5B words each.

**Status: needs exploring**

## [UberText](https://lang.org.ua/uk/corpora/#anchor4)

Corpus of scraped news articles from the several newspapers. They claim to have 6GB of data, zipped version has 1.6GB. The biggest problem is license. All the sentences are shuffled, so that the original text cannot be restored. Usage of the corpus is allowed only on the terms of **Fair Use** ([whatever that means](https://cedem.org.ua/analytics/fair-use/)). Will it be a problem if a model is later used in commercial purposes?

**Pros:**

* quite a lot of data
* data should be pretty high quality (news articles)
* tokenized (tokens and sentences)
* could be used for BERT's NSP (next sentence prediction) as negative examples (would it be biased?)

**Cons:**

* shuffled sentences!
* corpus contains wikipedia and literature too! (6GB contains them or not?)
* license

**Status: needs filtering and further exploring**

### Statistics

| part       | documents count* | tokens count |
|------------|-----------------:|-------------:|
| news       |       31.021.650 |  461.451.019 |
| wikipedia  |       15.786.948 |  185.645.357 |
| literature |        1.811.548 |   18.323.509 |
| whole      |       48.620.146 |  665.419.885 |

\* `document = sentence` - original documents have shuffled sentences, thus the average length of the result document is around 10-15 tokens

### Logs

#### v1.0

First view at the original data, artefacts:

* presence of really short sentences, some consist of numbers only, ex. `125180 69829` or `2012 Лондон`
* presence of some punctuation, like quotes
* some tokenization is done wrong, ex. `TwitterБукінгемського`, but that's the only one I have found so far
* some html present, ex. `3 4 2 style= color blue 5 45,0 23,00 27,0 32 style= text-align left Михайло Стефанськиий Дніпропетроська 2341` or `rowspan= 3 rowspan= 3 rowspan= 3`
* some sentences do not have Ukrainian, ex. `Cambridge Polity Press` or `''Stipa pennata'' subsp`
* some sentences have very little Ukrainian, ex. `Історія бриттів Historia Brittonum cum additamentis Nennii MGH AA T XII`
* markdown not cleared right?, ex. `**_НЕ МОЖУ СКАЗАТИ ЩО ОТОЧЕННЯ ЯНУКОВИЧА СТАЛО МОРАЛЬНІШИМ_` or `**_Корреспондент net_**`
* urls without punctuation, ex. `IFRAME 6 http www youtube com embed 3z9c9BDpl0s`
* words tokenized into single letters, ex. `С л а в' я т а здивований`
* metadata about pictures, etc, ex. `Інше Файл Do not buy Russian goods graffiti Kyiv jpg thumb 250px ...`, `File DP disk mag cartridge jpg Крупний план губок магазину`
* punctuation included for abbreviations, ex. `Сінкевич Олег Валерійович 30.07.73 р. н.`
* apostrophes removed, ex. `За його словами ці гроші досвід сприятимуть досягненню прогресу у розвязанні проблеми насильства в сім ї`

Reviewed up to `Карапакс широкий яйцеподібний опуклий на щитках карапаксу розташовані дрібні зморшки`.


## [Legal Corpus](https://lang.org.ua/uk/corpora/#anchor4)

Over 9GB of legal documents (laws and other things)

**Pros:**

* 9GB (quite a lot)
* specialized data
* quality data (should be at least)
* tokenized
* no license (i guess :))

**Cons:**

* specialized data :)

**Status: ???**

### Statistics

| documents count* | tokens count |
|------------------|--------------|
| 29.208.302       | 578.988.264  |

\* `document = sentence` - counted in paragraphs (legal documents divided into paragraphs)

### Logs

#### v1.0

First view at the original data, artefacts:

* a big number of documents are simply numbers of paragraphs
* some documents are simply names
* some words with hyphen are tokenized in the wrong way, the same words are sometimes tokenized in different ways, ex. `військово-технічної і військово- промислової політики та функціонування оборонно-промислового комплексу`, `щодо впровадження державно-приватного партнерства в оборонно- промисловому комплексі` or `з посиленою військово- фізичною підготовкою Кадетський корпус`
* some documents are really similar, ex.:
  * `Про призначення А Вовка державним уповноваженим Антимонопольного комітету України`
  * `Призначити ВОВКА Андрія Михайловича державним уповноваженим Антимонопольного комітету України`
  * `Про звільнення С Шершуна з посади державного уповноваженого Антимонопольного комітету України`
  * `Звільнити ШЕРШУНА Сергія Миколайовича з посади державного уповноваженого Антимонопольного комітету України`
* lots of duplicates, ex.:
  * `КАБІНЕТ МІНІСТРІВ УКРАЇНИ ПОСТАНОВА`
  * `А ЯЦЕНЮК`
  * `Прем'єр-міністр України`
  * `Президент України`
  * `Указ Президента України`
* some documents are really short, those are rather list points splitted into separate documents, ex.:
  * `Свічка запальна`
  * `Відро каністра для води`
  * `Лопата штикова`
* lots of names, ex.:
  * `ГРИЦЕНКО Павло Юхимович`
  * `ШЕВЧЕНКО Лариса Леонідівна`
  * `КОВАЛЕВСЬКА Тетяна Юріївна`
* some documents do not have Ukrainian, ex.:
  * `Vznz Ho х Um Umr х Kznzm Ugm Ugmr х Kznzgm Us х Ks Ugs х Kgs` - formula, all the words are some variables, which are described later
  * `січень СІЧ JAN лютий ЛЮТ FEB березень БЕР MAR квітень КВІ APR травень ТРА MAY червень ЧЕР JUN липень ЛИП JUL серпень СЕР AUG вересень ВЕР SEP жовтень ЖОВ OCT листопад ЛИС NOV грудень ГРУ DEC`

Reviewed up to `супроводження матеріальних цінностей і пасажирів у контрольованих зонах авіапідприємств аеропортах`.

## [CC-100](https://data.statmt.org/cc-100/)

Corpus which is created with the aim to replicate the training dataset for XLM-R. Ukrainian part contains 14GB of data. According to the `awesome-ukrainian-nlp` if we deduplicate the data we get around 10GB of it.

**Pros:**

* rather processed corpus
* diverse data (should be at least)
* quality data (same :))
* great license (as far as I understand)

**Cons:**

* not that much data for such a big corpus
* a lot of duplicates (?)

### Statistics

| documents count | tokens count |
|-----------------|--------------|
| ???             | ???          |

## [OSCAR](https://oscar-project.github.io/documentation/versions/oscar-2301/)

Version `23.01` contains 52GB data in Ukrainian.

**Pros:**

* a lot of data
* diverse data (should be at least)

**Cons:**

* probably a solid part of 52GB data is metadata and JSONL schema
* not very clear how do download it (just needs time to try)

**Status: waiting for the access approval**

### Statistics

| documents count | tokens count |
|-----------------|--------------|
| ???             | ???          |

## [mC4](https://github.com/allenai/allennlp/discussions/5056)

A big data corpus used for T5 training (i guess). According to `awesome-nlp-ukrainian` contains 196GB of data. Needs to be verified!

**Pros:**

* a lot of data (if everything is true, then it should be the main corpus)
* diverse data (should be at least)
* ok license (as far as I have understood)

**Cons:**

* not sure about the data quality
* duplicates (?)

**Status: needs a whole lot of filtering**

### Statistics

| documents count | tokens count                                     |
|-----------------|--------------------------------------------------|
| 38.556.465      | 17B estimated (haven't tokenized everything yet) |

### Logs

#### v1.0 (commit `554526e`)

First view at the tokenized data (tokenization using `spaCy` and simply omitting punctuation and whitespaces). Data is very dirty, I mean **very dirty**.

Artefacts:

* random greek letters, ex.: 
  * `evia top ασφαλιστικό πώς παίρνετε σύνταξη με λιγότερα ένσημα αρχειοθήκη ιστολογίου φεβ 25 103 φεβ 24 110 φεβ 23 102 φεβ 22 103 φεβ 21 96 φεβ 20 88 φεβ 19 91 φεβ 18`
* urls, ex.: 
  * `переклади без субтитрів магнітосфера землі з російської http://translate.thealphacentauri.net/book/255/1311`, `мапа війни у сирії українською syria.liveuamap.com/uk`
  * `постiйна адреса статтi http://svitlytsia.crimea.ua/?section=article&artid=14192`
  * `больной we try harder solder http://www.youtube.com/watch?v=sam4lq2whos&feature=player_embedded мізків у них немає тому що немає мізків у їхніх бітьків`
* hashtags, ex.: 
  * `assad#raqqa#r#syria news#syrian civil war#syrian civil war map#syria map`
* russian, ex.: 
  * `часть 28 плетение шнура часть 29 выкройка для вязаных изделий часть 35 вязание крючком с бусинами часть 36 скользящая петля`
  * `днепропетровская областная федерация пауэрлифтинга днепропетровская областная федерация пауэрлифтинга`
  * `гдз решебник по алгебре 7 класс мерзляк а.г. полонський в.б. рабінович ю.м. якір м.с. варіант 2 задание 160 2016 2017 онлайн`
  * `еще один человек умер после diablo 3 опубликовано 18.07.2012 gameway в новости игр52 тайваньский 18-летний подросток умер после двухдневной игры в diablo 3`
  * `наматрасник dreamline ппу 2 см 180х190 1800х1900 в воронеже недорого купить наматрасник размером 190х180 1900х1800 по низкой цене с доставкой главная наматрасники ортопедические ппу 2 см 180x190`
* html, ex.: 
  * `< a href="https://www.liveinternet.ru users/4434323 post194459080/">р‘рµр р·р ° рірѕр»рѕрірєр ° </a><br/>`
* weird characters, ex.: 
  * `єсѓр ° рѕр ° рљсѓр»сџ рќрµсѓрєрѕр»сњрєрѕ рѕрµсѓр»рѕр¶рѕс‹с рѕрї`
  * `ф¬zee÷kgcqpu0l¬zb÷2¬zy÷північна та центральна америка¬zc÷esnkwgd6¬zd÷p¬ze÷qbewlfnh¬zf÷0¬zo÷0¬zg÷2¬zh÷2_kgcqpu0l¬zj÷11¬zl÷/ua cricket north central america caribbean premier league/¬zx÷09північна та 030мерика0000000000001000карибська пр032лей оф000¬zcc÷0¬zaf÷північна та центральна америка¬~aa÷kipzly14¬ad÷1570914000¬ade÷1570`
* symbols?, ex.: 
  * `опубліковано 8.10.2012 16:48 | всі новини | версія для друку | коментарі 0 третина виборців не знає`
  * `дня соборності в олексіївській б ф № 3 пройшов захід для молоді`
  * `вигнутий смартфон виліковує свої подряпини → голосовий пошук від google`
  * `2009 2018 © екологія життя`
* emojis, ex.: 
  * `❗ ️ 5 із 6 сумських водіїв ігнорують ремені безпеки дослідження цукр онлайн журнал міста суми назад предыдущая запись 🎄 різдвяний ярмарок на трибуна парк готується до відкриття далее следующая запись 🗑 з’явився додаток`
  * `о плохом111 ну вот начался холивар 🙂 екзорцизм порча та прокляття реальність`
  * `освіта home ⇒ 📁 гдз з мови ⇒ займенники й контекст`
* wrong tokenization, ex:
  * `головнаглавная страницамеблікорпусні меблітумби`
* filenames, ex.:
  * `лотоцька с.к завірена.pdf 1).p7s ig-20134 001.pdf.p7s.p7s.p7s ig-20134-001.pdf 17 лютого 2020`
* inconsistent text, not connected, ex:
  * `мудак картонный и моем мнение не изменить аллах бабах тю які холівари є 2 реальності 1 ти живеш у присутності бога і ти це скрізь помічаєш та бачиш 2` - this seems to be from the YouTube comment section, and just showing all the comments
* slang, ex:
  * `быдлии на которую тонны народа фапают окааааай это такая знакомая ситуация` - what's more, it's russian
  * `звичайні дві дівахи переділись у варєте я тобі скажу чесно якщо кіба скаже харе я скажу ок і піду далі`
* swearing, ex:
  * `в коментах у єблі для мавп відносно дурки я там був примусово`
  * `поцьк 06.12.2017 20:36:11 ваше пахуеть`
* abbreviations, ex:
  * `першому головному лікарю приготуватися укмц usaid взаємодія`
  * `підм сам озн н. в ж р одн підм себе звор р. в дод їм ос д в множ дод собі зв м. в дод якийсь неознач ч р одн зн в означ ній особ м. в ж р одн обст` - syntax annotations
  * `карпати були точно сильнішими ніж фк львів`
* tags, ex:
  * `ніколи не здасть україни теги формула штайнмаєра новини грузії`
* surzhyk, ex:
  * `актьори які голівуд тіхо стогне в стороні пацани із дьоргающіміся сіськами це сільнєйший ріжисьорський ход з часів хіщніка і першого термінатора` 
  * `проклятиє укропи і молдаванє совмєстно заблокіровалі свінорускіє вайська в придністров ї коториє обєспєчівают мір і стабільность на украіно молдавской граніце ацкоє відео рекомендуємо к просмотру`

Interesting documents to monitor (need a better way of identification, than just a prefix):
  * `еще один человек умер после diablo 3 опубликовано 18.07.2012 gameway в новости игр52 тайваньский 18-летний подросток умер после двухдневной игры в diablo 3 как пишет the australian молодой человек 13 июля заказал в компьютерном клубе города тайнань отдельную кабинку и и
грал там в diablo 3 на протяжении двух дней без еды его обнаружили 15 июля лежачим на столе`
  * `petro_tazyk | entries tagged with ноу коментс entries tagged with ноу коментс nov 5th 2015 08:30 am jul 15th 2015 09:02 am уміють же знімати apr 9th 2015 03:03 pm і актьори які голівуд тіхо стогне в стороні пацани із дьоргающіміся сіськами це сільнєйший ріжисьорський ход з часів хіщніка і першого термінатора вухо нічо не ріже проклятиє укропи і молдаванє совмєстно заблокіровалі свінорускіє вайська в придністров ї коториє обєспєчівают мір і стабільность на украіно молдавской граніце ацкоє відео рекомендуємо к просмотру`

Reviewed up to `наматрасник dreamline ппу 2 см 180х190 1800х1900 в воронеже недорого купить наматрасник размером 190х180 1900х1800 по низкой цене с доставкой главная наматрасники ортопедические ппу 2 см 180x190`.

## TODO other corpora
