import re
import numpy as np
import pandas as pd

def cleartest(df, keywords):
    df2 = df['product_title']
    # 使用非捕获组消除警告
    pattern = r'\b(?:' + '|'.join([re.escape(kw) for kw in keywords]) + r')\b'
    aa = df2.str.contains(pattern, case=False, regex=True, na=False)

    # 获取匹配的行索引
    matching_indices = np.where(aa)[0]

    # 返回匹配的行和个数
    return df[aa], len(matching_indices)
keywords = [
    'pacifiers', 'baby', 'dummy',
    # 包含"奶嘴"的喂养用品
    "pacifier", "nipple", "baby bottle", "formula", "milk powder", "baby food",
    "bib", "sippy cup", "breast pump", "feeding bottle", "teething biscuits", "baby cereal",
    "bottle brush", "bottle sterilizer", "bottle warmer", "formula dispenser",
    "milk storage bags", "nursing cover", "nursing pillow", "burp cloths",
    "feeder", "weaning spoons", "baby bowls", "snack cup", "food grinder",
    "insulated bottle bag", "suckle", "anti-colic bottle", "transition cup",

    # 护理用品
    "diaper", "baby wipes", "baby powder", "baby lotion", "diaper cream", "thermometer",
    "nasal aspirator", "nail clippers", "baby brush", "grooming kit", "humidifier",
    "baby oil", "baby shampoo", "baby wash", "ear thermometer", "forehead thermometer",
    "diaper pail", "changing pad", "changing table", "potty", "training pants",
    "teething gel", "medicine dispenser", "baby first aid kit", "baby sunscreen",
    "insect repellent", "baby cotton buds", "baby soap", "baby detergent",

    # 衣物与纺织品
    "onesie", "swaddle", "baby blanket",
    "swaddle blanket", "muslin blanket", "receiving blanket",
    "quilt", "cot quilt", "baby quilt", "throw blanket",
    "security blanket", "comfort blanket",
    "thermal blanket", "waffle blanket",
    "baby sleep sack", "wearable blanket",
    "romper", "bodysuit", "pajamas", "sleepwear", "sleepsuit",
    "baby socks", "booties", "mittens", "hat", "beanie", "sun hat",
    "bib", "burp cloth", "hooded towel", "washcloth", "bath towel",
    "baby gown", "cardigan", "jacket", "snowsuit", "rain cover",
    "t-shirt", "pants", "leggings", "overalls", "dungarees",
    "swim diaper", "swimsuit", "rash guard", "swim vest",

    # 寝具与家具
    "crib", "bassinet", "playpen", "high chair", "baby monitor",
    "changing table", "dresser", "rocking chair", "glider", "nursery chair",
    "co-sleeper", "bedside sleeper", "toddler bed", "bed rail", "bed guard",
    "mattress", "mattress protector", "sheet", "fitted sheet", "crib bumper",
    "baby nest", "sleep pod", "baby hammock", "swing", "bouncer", "rocker",
    "play gym", "activity center", "walker", "jumper", "exercise saucer",
    "bookshelf", "toy storage", "laundry hamper", "nursery rug", "blackout curtains",

    # 出行用品
    "stroller", "car seat", "baby carrier",
    "pram", "travel system", "double stroller", "umbrella stroller",
    "jogging stroller", "stroller organizer", "stroller rain cover", "stroller fan",
    "stroller muff", "car seat canopy", "car seat protector", "car seat mirror",
    "baby wrap", "sling", "soft structured carrier", "hiking carrier", "backpack carrier",
    "diaper bag", "backpack diaper bag", "travel crib", "portable crib", "travel high chair",
    "car seat travel bag", "stroller travel bag", "luggage", "suitcase",

    # 洗浴用品
    "baby bathtub", "baby shampoo", "body wash", "towel", "bath toy", "rubber duck",
    "bath thermometer", "bath seat", "bath support", "bath sponge", "bath kneeler",
    "faucet cover", "bath rack", "soap dispenser", "shampoo rinse cup", "bath book",
    "bath crayons", "water toys", "splash mat", "bath robe", "hooded bath towel",
    "baby shower", "bath organizer", "non-slip mat", "toy net", "toy scoop",

    # 玩具与安抚
    "rattle", "teething toy", "teether", "stuffed animal", "mobile", "play mat",
    "soft book", "cloth book", "board book", "bath book", "activity mat", "gym mat",
    "play yard", "play tunnel", "tent", "ball pit", "balls", "stacking rings",
    "shape sorter", "nesting cups", "blocks", "building blocks", "pull toy",
    "push toy", "ride-on toy", "balance bike", "tricycle", "scooter",
    "musical toy", "xylophone", "drum", "piano", "music player", "white noise machine",
    "night light", "projector", "soother", "comfort item", "lovey", "cuddle toy",
    "plush toy", "doll", "action figure", "puzzle", "magnetic tiles", "train set",
    "play kitchen", "tool set", "doctor kit", "tea set", "play food",

    # 安全与健康
    "baby gate", "safety gate", "corner guard", "edge guard", "outlet cover",
    "cabinet lock", "door lock", "stove guard", "fireplace guard", "window guard",
    "monitor", "video monitor", "movement monitor", "breathing monitor", "wearable monitor",
    "air purifier", "dehumidifier", "cool mist humidifier", "warm mist humidifier",
    "baby scale", "growth chart", "health kit", "medicine syringe", "nasal spray",
    "vapor rub", "chest rub", "teething tablets", "gripe water", "probiotics",

    # 哺乳用品
    "nursing bra", "nursing tank", "nursing pads", "breast pads", "nipple shield",
    "nipple cream", "lanolin cream", "breast milk cooler", "cooler bag", "ice packs",
    "hands-free pumping bra", "pumping parts", "milk catcher", "haakaa", "let-down catcher",
    "nursing stool", "nursing necklace", "teething necklace", "amber necklace",

    # 特殊需求
    "weighted blanket", "sensory toy", "fidget toy", "chewelry", "chew necklace",
    "textured ball", "visual stimulation cards", "black and white toys", "high contrast toys",
    "adaptive spoon", "special needs bottle", "special needs chair", "therapy ball"
]
aa,bb = cleartest(df,keywords)
print(aa.head(20),bb)