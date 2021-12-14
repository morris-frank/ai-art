#%%
import io
import re
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from rich import print
from tqdm import trange

uri="https://ape.offbase.org"
out = Path("./apes/")
out.mkdir(exist_ok=True)
max_idx = 10_000

# %%
# Download all the Apes
def get_ape(idx: int):
    html: str = requests.get(f"{uri}/token/{idx}").text
    img_link = re.search("\/ipfs\/.*\"", html)[0][6:-1]
    img_bytes = requests.get(f"{uri}{img_link}").content
    img = Image.open(io.BytesIO(img_bytes))
    table = re.findall(r"<table>(.*?)</table>", html, flags=re.DOTALL)[0]
    desc = " ".join(re.sub(r'<[^>]*>', ' ', table).replace("\n", " ").split())

    img.save(str(out / f"{i:04}.png"))
    with open("apes.eng.csv", "a") as fp:
        fp.write(f"\n{i},{i:04}.png,{desc}")

for i in trange(0, max_idx):
    try:
        get_ape(i)
    except:
        print(f"Couldnt do {i}")
        pass
        
#%%
# Translate the text annotations to Russian
df = pd.read_csv("./apes.eng.csv")
df["caption"] = df['caption'].str.replace("&#39;", "")
pd.set_option('display.max_colwidth', None)
translation_map = {
    "1": "1",
    "2": "2",
    "3d": "3d",
    "Admirals": "Адмиралы",
    "Angry": "Angry",
    "Angry": "Сердитый",
    "Aquamarine": "Аквамарин",
    "Army": "Армия",
    "Baby39s": "Малыши",
    "Babys": "Дети",
    "Background": "Фон",
    "Bandana": "Бандана",
    "Bandolier": "Бандольер",
    "Bayc": "Bayc",
    "Beams": "Лучи",
    "Beanie": "Бини",
    "Biker": "Байкер",
    "Black": "Черный",
    "Blindfold": "Повязка на глаза",
    "Bloodshot": "Кровавый ожог",
    "Bloodshot": "Кровосток",
    "Blue": "Синий",
    "Boho": "Бохо",
    "Bone": "Кость",
    "Bonnet": "Боннет",
    "Bored": "Скучно",
    "Bot": "Бот",
    "Bowler": "Боулер",
    "Bowler": "Котелок",
    "Brim": "Окантовка",
    "Brown": "Коричневый",
    "Bubblegum": "Bubblegum",
    "Bubblegum": "Бубльгум",
    "Bunny": "Кролик",
    "Captain39s": "Капитан39s",
    "Captains": "Капитаны",
    "Caveman": "Пещерный человек",
    "Cheetah": "Гепард",
    "Chef": "Шеф-повар",
    "Cigar": "Сигара",
    "Cigarette": "Сигарета",
    "Closed": "Закрытый",
    "Clothes": "Одежда",
    "Coat": "Пальто",
    "Coins": "Монеты",
    "Commie": "Коммунист",
    "Cowboy": "Ковбой",
    "Crazy": "Сумасшедший",
    "Cream": "Крем",
    "Cross": "Крест",
    "Crown": "Корона",
    "Cyborg": "Киборг",
    "Dagger": "Кинжал",
    "Dark": "Темный",
    "Death": "Смерть",
    "Diamond": "Алмаз",
    "Diamond": "Бриллиант",
    "Discomfort": "Дискомфорт",
    "Dmt": "Dmt",
    "Dress": "Платье",
    "Dumbfounded": "Ошарашенный",
    "Dumbfounded": "Ошеломленный",
    "Dye": "Dye",
    "Dye": "Краситель",
    "Earring": "Серьга",
    "Ears": "Уши",
    "Era": "Эра",
    "Eyed": "Глазастый",
    "Eyepatch": "Повязка для глаз",
    "Eyepatch": "Повязка на глаза",
    "Eyes": "Глаза",
    "Faux": "Faux",
    "Faux": "Фальшивые",
    "Fez": "Фес",
    "Fisherman39s": "Рыбаки",
    "Fishermans": "Рыбаки",
    "Flipped": "Перевернутый",
    "Fur": "Мех",
    "Girl39s": "Девушки39",
    "Girls": "Девочки",
    "Gold": "Золото",
    "Golden": "Золото",
    "Golden": "Золотой",
    "Gray": "Серый",
    "Green": "Зеленый",
    "Grill": "Гриль",
    "Grin": "Grin",
    "Grin": "Гриль",
    "Guayabera": "Гуаябера",
    "Hair": "Волосы",
    "Halo": "Halo",
    "Halo": "Ореол",
    "Hat": "Шляпа",
    "Hawaiian": "Гавайи",
    "Hawaiian": "Гавайский",
    "Hawk": "Ястреб",
    "Headband": "Ободок",
    "Headband": "Повязка на голову",
    "Heart": "Сердце",
    "Helm": "Шлем",
    "Helmet": "Шлем",
    "Hip": "Бедро",
    "Hip": "Хип",
    "Holes": "Дырки",
    "Holes": "Отверстия",
    "Holographic": "Голографический",
    "Hoop": "Обруч",
    "Hop": "Хмель",
    "Hop": "Хоп",
    "horn": "рог",
    "Horn": "Рог",
    "Horns": "Рога",
    "Hypnotized": "Загипнотизированный",
    "Irish": "Irish",
    "Irish": "Ирландский",
    "Jacket": "Пиджак",
    "Jovial": "Веселый",
    "Jumpsuit": "Джемпер",
    "Jumpsuit": "Костюм",
    "Kazoo": "Kazoo",
    "Kazoo": "Казу",
    "King39s": "King39s",
    "Kings": "Короли",
    "L": "L",
    "Lab": "Лаборатория",
    "Laser": "Лазер",
    "Laurel": "Лавр",
    "Laurel": "Лорел",
    "Leather": "Кожа",
    "Logo": "Логотип",
    "Lumberjack": "Lumberjack",
    "Lumberjack": "Дровосек",
    "Mohawk": "Ирокез",
    "Motorcycle": "Мотоцикл",
    "Mouth": "Рот",
    "Multicolored": "Многоцветный",
    "Multicolored": "Разноцветный",
    "Navy": "ВМФ",
    "Necklace": "Ожерелье",
    "New": "Новый",
    "Noise": "Шум",
    "Oh": "О",
    "ooo": "ooo",
    "Orange": "Оранжевый",
    "Out": "Out",
    "Party": "Вечеринка",
    "Pelt": "Шкурка",
    "Phoneme": "Фонема",
    "Pilot": "Пилот",
    "Pimp": "Сутенер",
    "Pink": "Розовый",
    "Pipe": "Труба",
    "Pizza": "Пицца",
    "Police": "Полиция",
    "Prison": "Тюрьма",
    "Prom": "Выпускной",
    "Prom": "Пром",
    "Prussian": "Прусс",
    "Prussian": "Пруссия",
    "Puffy": "Puffy",
    "Puffy": "Пухлый",
    "Punk": "Панк",
    "Purple": "Пурпурный",
    "Purple": "Фиолетовый",
    "Rage": "Ярость",
    "Rainbow": "Радуга",
    "Red": "Красный",
    "Robe": "Халат",
    "Robot": "Робот",
    "Sad": "Грустный",
    "Sad": "Грусть",
    "Safari": "Сафари",
    "Sailor": "Моряк",
    "Sampm": "Sampm",
    "Scumbag": "Отморозок",
    "Sea": "Море",
    "Seaman39s": "моряк",
    "Seamans": "Моряки",
    "Service": "Сервис",
    "Service": "Служба",
    "Shirt": "Рубашка",
    "Short": "Короткий",
    "Silver": "Серебро",
    "Sleepy": "Сонник",
    "Sleepy": "Сонный",
    "Sleeveless": "Без рукавов",
    "Sleeveless": "Безрукавка",
    "Small": "Маленький",
    "Smoking": "Курение",
    "Solid": "Сплошной",
    "Solid": "Твердый",
    "Space": "Космос",
    "Spinner": "Спиннер",
    "Striped": "Полосатый",
    "Stud": "Шпилька",
    "Stunt": "Каскадер",
    "Stunt": "Трюк",
    "Stuntman": "Каскадер",
    "Suit": "Костюм",
    "Sunglasses": "Солнцезащитные очки",
    "Sushi": "Суши",
    "Suspenders": "Подтяжки",
    "T": "T",
    "Tan": "Загар",
    "Tanktop": "Танкотоп",
    "Tanktop": "Танктоп",
    "Tee": "Ти",
    "Tee": "Футболка",
    "Tie": "Галстук",
    "Toga": "Тога",
    "Tongue": "Язык",
    "Trippy": "Trippy",
    "Trippy": "Триппи",
    "Turtleneck": "Водолазка",
    "Tuxedo": "Смокинг",
    "Tweed": "Твид",
    "Unshaven": "Небритый",
    "Vest": "Жилет",
    "Vietnam": "Вьетнам",
    "Vuh": "Вух",
    "Wah": "Вах",
    "White": "Белый",
    "Wide": "Широкий",
    "Wool": "Шерсть",
    "Work": "Работа",
    "Wreath": "Венок",
    "Ww2": "Ww2",
    "X": "X",
    "Yellow": "Желтый",
    "Zombie": "Зомби",
}

words = set(re.sub('[^\w ]+','', df["caption"].to_string(index=False).replace("\n", " ")).split(" "))
for word in filter(lambda w: not w in translation_map, words):
    print(word)

for eng, rus in translation_map.items():
    df["caption"] = df['caption'].str.replace(eng, rus)
df.to_csv("apes.csv")