"""
generate_data.py
-----------------
Creates a curated dataset of top English and Malayalam songs with
realistic Spotify audio features, metadata, album covers, and synthetic
user listening history for collaborative filtering.
"""

from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

# Core curated tracks (49 Malayalam + 50 English; additional tracks follow below)
MALAYALAM_TRACKS = [
    # Happy / Dance / Upbeat
    ("Jimikki Kammal", "ജിമിക്കി കമ്മൽ", "Vineeth Sreenivasan, Renjith Unni", "Velipadinte Pusthakam", "Happy / Dance", "Happy", 0.88, 0.90, 0.92, 0.10, 122.0, -4.5, 92, "3:30"),
    ("Kudukku", "കുടുക്ക്", "Vineeth Sreenivasan", "Love Action Drama", "Pop / Dance", "Happy", 0.86, 0.88, 0.90, 0.12, 120.0, -5.0, 90, "3:15"),
    ("Aluva Puzhayude", "ആലുവ പുഴയുടെ", "Vineeth Sreenivasan", "Premam", "Acoustic / Melody", "Happy", 0.72, 0.68, 0.85, 0.45, 108.0, -7.0, 94, "3:40"),
    ("Illuminati", "ഇല്ലുമിനാറ്റി", "Sushin Shyam, Dabzee", "Aavesham", "Hip-Hop / Party", "Party", 0.88, 0.94, 0.88, 0.05, 128.0, -3.8, 98, "3:12"),
    ("Aalroopangal", "ആൾരൂപങ്ങൾ", "Vishnu Vijay", "Premalu", "Pop / Upbeat", "Happy", 0.80, 0.82, 0.86, 0.18, 116.0, -5.2, 91, "3:22"),
    ("Welcome to Hyderabad", "വെൽക്കം ടു ഹൈദരാബാദ്", "Sanjith Hegde", "Premalu", "Party / Fun", "Party", 0.86, 0.88, 0.88, 0.12, 122.0, -4.8, 89, "3:35"),
    ("Mini Maharani", "മിനി മഹാറാണി", "Kapil Kapilan", "Premalu", "Romantic / Playful", "Happy", 0.78, 0.74, 0.82, 0.22, 110.0, -5.8, 90, "3:08"),
    ("Kanakam Moolam", "കനകം മൂലം", "Vineeth Sreenivasan", "Kanakam Kaamini Kalaham", "Playful / Fun", "Happy", 0.76, 0.72, 0.84, 0.25, 112.0, -6.0, 80, "3:18"),
    ("Devadoothar Paadi", "ദേവദൂതർ പാടി", "Biju Narayanan", "Nna Thaan Case Kodu", "Retro / Dance", "Party", 0.84, 0.88, 0.92, 0.15, 120.0, -5.0, 93, "3:45"),

    # Romantic / Love
    ("Malare", "മലരേ", "Vijay Yesudas, Rajesh Murugesan", "Premam", "Melody / Romance", "Romantic", 0.60, 0.55, 0.78, 0.65, 96.0, -7.5, 98, "4:42"),
    ("Aaradhike", "ആരാധികേ", "Sooraj Santhosh, Madhuvanthi", "Ambili", "Melody / Soul", "Romantic", 0.65, 0.60, 0.82, 0.50, 104.0, -6.8, 97, "4:10"),
    ("Darshana", "ദർശനാ", "Hesham Abdul Wahab", "Hridayam", "Romantic / Pop", "Romantic", 0.70, 0.78, 0.75, 0.30, 118.0, -5.5, 96, "3:58"),
    ("Uyire", "ഉയിരേ", "Narayani Gopan, Shaan Rahman", "Minnal Murali", "Melody / Romance", "Romantic", 0.58, 0.52, 0.65, 0.70, 92.0, -8.0, 94, "4:15"),
    ("Pavizha Mazha", "പവിഴ മഴ", "K.S. Harisankar, Jakes Bejoy", "Athiran", "Romantic / Acoustic", "Romantic", 0.62, 0.58, 0.68, 0.60, 98.0, -7.2, 93, "4:05"),
    ("Jeevamshamayi", "ജീവാംശമായി", "K.S. Harisankar, Shreya Ghoshal", "Theevandi", "Melody / Romance", "Romantic", 0.55, 0.50, 0.72, 0.68, 90.0, -8.2, 95, "5:12"),
    ("Uyiril Thodum", "ഉയിരിൽ തൊടും", "Sooraj Santhosh, Anne Amie", "Kumbalangi Nights", "Romantic / Chill", "Romantic", 0.52, 0.48, 0.70, 0.75, 88.0, -8.8, 94, "3:52"),
    ("Neela Nilave", "നീല നിലാവേ", "Kapil Kapilan, Jakes Bejoy", "RDX", "Romantic / Dance", "Romantic", 0.76, 0.80, 0.82, 0.25, 112.0, -5.9, 95, "4:12"),
    ("Lailakame", "ലൈലാകമേ", "Hesham Abdul Wahab, Haricharan", "Hridayam", "Romantic / Soulful", "Romantic", 0.56, 0.58, 0.70, 0.62, 94.0, -7.8, 91, "4:20"),
    ("Kinavile", "കിനാവിലെ", "Hesham Abdul Wahab", "Pranaya Vilasam", "Romantic / Chill", "Romantic", 0.62, 0.54, 0.74, 0.58, 98.0, -7.9, 88, "3:40"),
    ("Mazhaye Thoomanje", "മഴയെ തൂമഞ്ഞേ", "Haricharan, Mridula Warrier", "Pattam Pole", "Romantic / Rainy", "Romantic", 0.55, 0.52, 0.68, 0.65, 92.0, -8.1, 87, "4:30"),
    ("Pookkal Panineer Pookkal", "പൂക്കൾ പനിനീർ പൂക്കൾ", "K.J. Yesudas, Vani Jayaram", "Action Hero Biju", "Classic / Melody", "Romantic", 0.50, 0.48, 0.70, 0.75, 90.0, -8.5, 86, "4:18"),

    # Sad / Melancholic / Heartbroken
    ("Mizhiyil Ninnum", "മിഴിയിൽ നിന്നും", "Shahabaz Aman", "Mayaanadhi", "Soul / Sad", "Sad", 0.38, 0.32, 0.28, 0.82, 82.0, -11.0, 93, "4:40"),
    ("Thaniye", "തനിയേ", "Sooraj Santhosh", "Guppy", "Acoustic / Sad", "Sad", 0.30, 0.28, 0.22, 0.88, 78.0, -12.0, 92, "4:25"),
    ("Akale", "അകലെ", "Haricharan, Anne Amie", "9", "Melancholic", "Heartbroken", 0.34, 0.35, 0.25, 0.78, 80.0, -10.5, 88, "4:12"),
    ("Azhalinte Aazhangalil", "അഴലിന്റെ ആഴങ്ങളിൽ", "Nikhil Mathew", "Ayalum Njanum Thammil", "Emotional / Sad", "Heartbroken", 0.28, 0.30, 0.18, 0.84, 76.0, -12.2, 91, "5:02"),
    ("Neeyo", "നീയോ", "Hesham Abdul Wahab", "Hridayam", "Sad / Reflective", "Heartbroken", 0.40, 0.45, 0.35, 0.72, 88.0, -9.2, 86, "4:00"),
    ("Kanne Kanne", "കണ്ണേ കണ്ണേ", "Hesham Abdul Wahab", "Animal (Malayalam)", "Sad / Emotional", "Heartbroken", 0.35, 0.42, 0.26, 0.70, 82.0, -10.0, 87, "3:50"),
    ("Enthe Innum Vanneela", "എന്തേ ഇന്നും വന്നീല", "K.J. Yesudas, K.S. Chithra", "Gramaphone", "Classical / Sad", "Sad", 0.24, 0.26, 0.20, 0.92, 74.0, -13.0, 85, "4:50"),
    ("Vannathi", "വണ്ണാത്തി", "Haricharan", "Ezra", "Melancholy / Love", "Sad", 0.42, 0.48, 0.45, 0.72, 86.0, -9.5, 84, "3:48"),

    # Chill / Relaxed / Acoustic
    ("Cherathukal", "ചെരാതുകൾ", "Sushin Shyam, Sithara", "Kumbalangi Nights", "Indie / Acoustic", "Chill", 0.42, 0.35, 0.48, 0.85, 86.0, -10.8, 96, "3:42"),
    ("Parudeesa", "പറുദീസാ", "Sreenath Bhasi, Sushin Shyam", "Bheeshma Parvam", "Chill / Groove", "Chill", 0.74, 0.66, 0.62, 0.35, 102.0, -7.0, 94, "3:25"),
    ("Kaantha", "കാന്താ", "Sushin Shyam", "Kumbalangi Nights", "Folk / Ambient", "Chill", 0.48, 0.40, 0.55, 0.80, 85.0, -10.2, 89, "3:10"),
    ("Kaathodu Kaathorom", "കാതോട് കാതോരം", "Job Kurian", "Independent Indie", "Acoustic / Soul", "Chill", 0.44, 0.42, 0.58, 0.82, 84.0, -10.0, 86, "4:15"),
    ("Kadhale Kadhale", "കാതലേ കാതലേ", "Govind Vasantha", "Melody Single", "Acoustic / Nostalgic", "Chill", 0.36, 0.38, 0.42, 0.86, 76.0, -11.0, 90, "3:30"),
    ("Ente Ummante Peru", "എന്റെ ഉമ്മാന്റെ പേര്", "Sreenath Bhasi", "Ente Ummante Peru", "Groove / Relaxed", "Chill", 0.66, 0.58, 0.65, 0.45, 98.0, -8.0, 82, "3:20"),
    ("Maanam Thudukkanu", "മാനം തുടുക്കണു", "Karthik", "Varane Avashyamund", "Feel-good / Breeze", "Chill", 0.68, 0.64, 0.80, 0.40, 106.0, -7.2, 88, "3:50"),

    # Energetic / Workout / Hype
    ("Jaada", "ജാഡ", "Sushin Shyam", "Aavesham", "Rap / Bass", "Energetic", 0.82, 0.95, 0.82, 0.04, 135.0, -3.5, 97, "3:05"),
    ("Armadham", "അർമാദം", "Sushin Shyam, Pranav CP", "Aavesham", "EDM / Trap", "Energetic", 0.88, 0.94, 0.86, 0.05, 132.0, -3.7, 96, "3:18"),
    ("Scene Mone", "സീൻ മോനെ", "Sushin Shyam", "Aavesham", "Rap / Hype", "Party", 0.89, 0.96, 0.85, 0.03, 130.0, -3.2, 95, "2:55"),
    ("Manavalan Thug", "മണവാളൻ തഗ്ഗ്", "Dabzee, SA", "Thallumaala", "Hip-Hop / Street", "Party", 0.88, 0.90, 0.84, 0.06, 124.0, -4.2, 94, "3:10"),
    ("Ole Melody", "ഓലെ മെലഡി", "Vishnu Vijay, Haricharan", "Thallumaala", "Dance / Pop", "Happy", 0.80, 0.78, 0.80, 0.20, 116.0, -5.8, 92, "3:25"),
    ("Pala Palli Thiruppalli", "പാലാ പള്ളി തിരുപ്പള്ളി", "Jakes Bejoy, Atul Narukara", "Kaduva", "Folk / High-Energy", "Party", 0.86, 0.94, 0.90, 0.08, 126.0, -4.0, 93, "3:40"),
    ("Eyy Banane", "എയ്യ് ബനാനെ", "Neha Nair, Jakes Bejoy", "RDX", "Dance / Beat", "Party", 0.82, 0.86, 0.82, 0.15, 120.0, -5.2, 90, "3:15"),
    ("Rathipushpam", "രതിപുഷ്പം", "Unni Menon", "Bheeshma Parvam", "Disco / Retro", "Party", 0.80, 0.74, 0.78, 0.20, 114.0, -6.1, 91, "4:00"),

    # Peaceful / Soulful / Focus
    ("Oru Rathri Koodi", "ഒരു രാത്രി കൂടി", "K.J. Yesudas, K.S. Chithra", "Summer in Bethlehem", "Classic / Soulful", "Soulful", 0.32, 0.36, 0.40, 0.85, 80.0, -11.0, 91, "4:55"),
    ("Aararum Kaanathe", "ആരാരും കാണാതെ", "P. Jayachandran", "Chandrolsavam", "Soul / Melody", "Soulful", 0.35, 0.38, 0.50, 0.88, 78.0, -11.5, 89, "4:35"),
    ("Kannodu Kannidayum", "കണ്ണോട് കണ്ണിടയും", "K.J. Yesudas", "Kaathodu Kaathoram", "Classic / Nostalgic", "Soulful", 0.40, 0.42, 0.60, 0.84, 82.0, -10.2, 88, "4:20"),
    ("Nilapakshi", "നീലപ്പക്ഷി", "K.S. Chithra, Harisankar", "Marakkar", "Melody / Spiritual", "Soulful", 0.38, 0.44, 0.52, 0.80, 84.0, -9.5, 86, "4:40"),
    ("Ennavale", "എന്നവളേ", "Unni Menon", "Nostalgic Classics", "Soul / Melody", "Soulful", 0.44, 0.46, 0.62, 0.78, 86.0, -9.0, 88, "4:15"),
]

ENGLISH_TRACKS = [
    # Happy / Dance / Feel-Good
    ("Happy", "Happy", "Pharrell Williams", "G I R L", "Pop / Funk", "Happy", 0.86, 0.85, 0.95, 0.12, 120.0, -5.0, 95, "3:53"),
    ("Can't Stop the Feeling!", "Can't Stop the Feeling!", "Justin Timberlake", "Trolls OST", "Dance / Pop", "Happy", 0.84, 0.88, 0.92, 0.08, 122.0, -4.8, 93, "3:56"),
    ("Shape of You", "Shape of You", "Ed Sheeran", "÷ (Divide)", "Pop / Dancehall", "Happy", 0.88, 0.80, 0.86, 0.15, 122.0, -5.2, 98, "3:53"),
    ("Levitating", "Levitating", "Dua Lipa", "Future Nostalgia", "Disco / Pop", "Happy", 0.88, 0.86, 0.90, 0.06, 124.0, -4.5, 96, "3:23"),
    ("Uptown Funk", "Uptown Funk", "Mark Ronson ft. Bruno Mars", "Uptown Special", "Funk / Pop", "Party", 0.90, 0.92, 0.94, 0.02, 126.0, -3.9, 97, "4:30"),
    ("Shake It Off", "Shake It Off", "Taylor Swift", "1989", "Pop / Dance", "Happy", 0.86, 0.88, 0.90, 0.05, 122.0, -4.2, 96, "3:39"),
    ("Don't Stop Me Now", "Don't Stop Me Now", "Queen", "Jazz", "Classic Rock / Anthem", "Happy", 0.74, 0.92, 0.92, 0.10, 130.0, -4.0, 94, "3:29"),
    ("Here Comes the Sun", "Here Comes the Sun", "The Beatles", "Abbey Road", "Classic / Acoustic", "Happy", 0.58, 0.52, 0.88, 0.65, 98.0, -8.2, 95, "3:05"),
    ("Flowers", "Flowers", "Miley Cyrus", "Endless Summer Vacation", "Disco / Pop", "Happy", 0.82, 0.80, 0.82, 0.14, 118.0, -5.1, 95, "3:20"),
    ("Something Just Like This", "Something Just Like This", "The Chainsmokers & Coldplay", "Memories...Do Not Open", "EDM / Pop", "Happy", 0.75, 0.82, 0.74, 0.12, 120.0, -5.0, 93, "4:07"),

    # Romantic / Love
    ("Perfect", "Perfect", "Ed Sheeran", "÷ (Divide)", "Acoustic / Romance", "Romantic", 0.52, 0.45, 0.72, 0.80, 95.0, -8.0, 98, "4:23"),
    ("Until I Found You", "Until I Found You", "Stephen Sanchez", "Easy on My Eyes", "Retro / Romance", "Romantic", 0.54, 0.52, 0.70, 0.65, 94.0, -7.8, 96, "2:57"),
    ("All of Me", "All of Me", "John Legend", "Love in the Future", "Piano / Soul", "Romantic", 0.42, 0.38, 0.65, 0.88, 80.0, -8.8, 96, "4:29"),
    ("Lover", "Lover", "Taylor Swift", "Lover", "Acoustic / Indie Pop", "Romantic", 0.60, 0.50, 0.75, 0.70, 92.0, -7.9, 95, "3:41"),
    ("Golden Hour", "Golden Hour", "JVKE", "This Is What Feels Like", "Orchestral Pop", "Romantic", 0.52, 0.55, 0.70, 0.68, 96.0, -7.2, 94, "3:29"),
    ("Just the Way You Are", "Just the Way You Are", "Bruno Mars", "Doo-Wops & Hooligans", "Pop / Romance", "Romantic", 0.72, 0.82, 0.85, 0.18, 116.0, -5.3, 94, "3:40"),
    ("Thinking Out Loud", "Thinking Out Loud", "Ed Sheeran", "x", "Blue-eyed Soul", "Romantic", 0.62, 0.46, 0.72, 0.68, 88.0, -8.1, 95, "4:41"),
    ("Say You Won't Let Go", "Say You Won't Let Go", "James Arthur", "Back from the Edge", "Acoustic / Pop", "Romantic", 0.55, 0.48, 0.68, 0.74, 90.0, -8.0, 93, "3:31"),
    ("Photograph", "Photograph", "Ed Sheeran", "x", "Acoustic / Nostalgic", "Romantic", 0.50, 0.48, 0.60, 0.72, 86.0, -8.5, 92, "4:19"),
    ("As It Was", "As It Was", "Harry Styles", "Harry's House", "Synth-Pop / Melody", "Romantic", 0.78, 0.74, 0.76, 0.32, 118.0, -6.1, 97, "2:47"),

    # Sad / Melancholic / Heartbroken
    ("Someone Like You", "Someone Like You", "Adele", "21", "Piano / Soul", "Sad", 0.35, 0.32, 0.20, 0.90, 82.0, -11.0, 97, "4:45"),
    ("When I Was Your Man", "When I Was Your Man", "Bruno Mars", "Unorthodox Jukebox", "Piano / Ballad", "Heartbroken", 0.36, 0.34, 0.25, 0.90, 80.0, -11.2, 95, "3:33"),
    ("Fix You", "Fix You", "Coldplay", "X&Y", "Alt-Rock / Anthem", "Sad", 0.32, 0.42, 0.24, 0.78, 84.0, -10.2, 96, "4:55"),
    ("Glimpse of Us", "Glimpse of Us", "Joji", "SMITHEREENS", "Piano / Melancholy", "Heartbroken", 0.28, 0.25, 0.15, 0.92, 76.0, -12.5, 95, "3:53"),
    ("Drivers License", "Drivers License", "Olivia Rodrigo", "SOUR", "Bedroom Pop / Sad", "Heartbroken", 0.38, 0.45, 0.22, 0.76, 86.0, -10.0, 96, "4:02"),
    ("Let Her Go", "Let Her Go", "Passenger", "All the Little Lights", "Folk / Acoustic", "Sad", 0.45, 0.40, 0.30, 0.82, 88.0, -9.2, 94, "4:12"),
    ("All I Want", "All I Want", "Kodaline", "In a Perfect World", "Indie Rock / Sad", "Heartbroken", 0.30, 0.38, 0.18, 0.82, 78.0, -11.5, 91, "5:05"),
    ("See You Again", "See You Again", "Wiz Khalifa ft. Charlie Puth", "Furious 7 OST", "Hip-Hop / Tribute", "Sad", 0.58, 0.52, 0.35, 0.60, 88.0, -8.2, 95, "3:49"),

    # Chill / Relaxed / Coffee / Acoustic
    ("Sunflower", "Sunflower", "Post Malone & Swae Lee", "Spider-Man OST", "Chill Hip-Hop", "Chill", 0.76, 0.62, 0.84, 0.40, 90.0, -6.8, 97, "2:38"),
    ("Sunday Morning", "Sunday Morning", "Maroon 5", "Songs About Jane", "Soul / Chill", "Chill", 0.68, 0.55, 0.78, 0.45, 95.0, -7.2, 92, "4:02"),
    ("Banana Pancakes", "Banana Pancakes", "Jack Johnson", "In Between Dreams", "Acoustic / Relaxed", "Chill", 0.60, 0.35, 0.80, 0.85, 88.0, -10.1, 91, "3:12"),
    ("Slow Dancing in a Burning Room", "Slow Dancing in a Burning Room", "John Mayer", "Continuum", "Blues / Rock", "Chill", 0.48, 0.48, 0.45, 0.68, 86.0, -9.0, 93, "4:02"),
    ("Heat Waves", "Heat Waves", "Glass Animals", "Dreamland", "Psychedelic Pop", "Chill", 0.75, 0.66, 0.68, 0.30, 104.0, -6.2, 96, "3:58"),
    ("Sunset Lover", "Sunset Lover", "Petit Biscuit", "Presence", "Chill Electronic", "Chill", 0.72, 0.58, 0.75, 0.42, 100.0, -7.0, 92, "3:57"),
    ("Anti-Hero", "Anti-Hero", "Taylor Swift", "Midnights", "Synth-Pop", "Chill", 0.78, 0.72, 0.70, 0.20, 114.0, -6.2, 95, "3:20"),

    # Energetic / Workout / Hype
    ("Blinding Lights", "Blinding Lights", "The Weeknd", "After Hours", "Synthwave / Pop", "Energetic", 0.82, 0.88, 0.78, 0.05, 130.0, -4.2, 98, "3:20"),
    ("Believer", "Believer", "Imagine Dragons", "Evolve", "Alt-Rock / Trap", "Energetic", 0.76, 0.92, 0.68, 0.04, 132.0, -3.2, 97, "3:24"),
    ("Eye of the Tiger", "Eye of the Tiger", "Survivor", "Eye of the Tiger", "Hard Rock / Workout", "Energetic", 0.70, 0.94, 0.74, 0.02, 134.0, -3.0, 94, "4:05"),
    ("Till I Collapse", "Till I Collapse", "Eminem", "The Eminem Show", "Hip-Hop / Motivation", "Energetic", 0.78, 0.95, 0.60, 0.03, 136.0, -2.8, 96, "4:57"),
    ("Stay", "Stay", "The Kid LAROI & Justin Bieber", "F*CK LOVE 3", "Pop / Trap", "Energetic", 0.80, 0.86, 0.78, 0.08, 128.0, -4.3, 96, "2:21"),
    ("Wake Me Up", "Wake Me Up", "Avicii", "True", "EDM / Folktronica", "Energetic", 0.78, 0.90, 0.80, 0.10, 128.0, -3.8, 95, "4:07"),
    ("Counting Stars", "Counting Stars", "OneRepublic", "Native", "Pop Rock / Anthem", "Energetic", 0.78, 0.88, 0.72, 0.08, 128.0, -4.1, 94, "4:17"),
    ("Viva La Vida", "Viva La Vida", "Coldplay", "Viva la Vida", "Baroque Pop / Anthem", "Energetic", 0.65, 0.82, 0.70, 0.20, 120.0, -5.4, 95, "4:02"),

    # Party / Dance
    ("Starboy", "Starboy", "The Weeknd ft. Daft Punk", "Starboy", "Electro-R&B", "Party", 0.82, 0.85, 0.74, 0.06, 126.0, -4.1, 96, "3:50"),
    ("Don't Start Now", "Don't Start Now", "Dua Lipa", "Future Nostalgia", "Nu-Disco", "Party", 0.88, 0.90, 0.88, 0.04, 124.0, -3.9, 95, "3:03"),
    ("Bad Guy", "Bad Guy", "Billie Eilish", "WHEN WE ALL FALL ASLEEP...", "Alt-Pop / Bass", "Party", 0.85, 0.72, 0.65, 0.22, 115.0, -6.0, 95, "3:14"),

    # Peaceful / Soulful / Focus
    ("Weightless", "Weightless", "Marconi Union", "Ambient Transmissions", "Ambient / Drone", "Soulful", 0.15, 0.12, 0.30, 0.98, 70.0, -18.0, 90, "8:00"),
    ("River Flows in You", "River Flows in You", "Yiruma", "First Love", "Neo-Classical Piano", "Soulful", 0.28, 0.20, 0.45, 0.95, 75.0, -15.0, 93, "3:08"),
    ("Clair de Lune", "Clair de Lune", "Claude Debussy", "Suite Bergamasque", "Classical", "Soulful", 0.20, 0.15, 0.35, 0.99, 68.0, -16.5, 92, "5:03"),
    ("Experience", "Experience", "Ludovico Einaudi", "In a Time Lapse", "Neo-Classical", "Soulful", 0.25, 0.50, 0.40, 0.90, 90.0, -10.2, 94, "5:15"),
]

# Curated High-Quality Music Cover Artworks (Unsplash / Spotify Aesthetic Music Covers)
AESTHETIC_COVERS = {
    "Happy": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=400&auto=format&fit=crop&q=80",
    "Romantic": "https://images.unsplash.com/photo-1518199266791-5375a83190b7?w=400&auto=format&fit=crop&q=80",
    "Sad": "https://images.unsplash.com/photo-1518495973542-4542c06a5843?w=400&auto=format&fit=crop&q=80",
    "Heartbroken": "https://images.unsplash.com/photo-1499209974431-9dddcece7f88?w=400&auto=format&fit=crop&q=80",
    "Chill": "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=400&auto=format&fit=crop&q=80",
    "Energetic": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&auto=format&fit=crop&q=80",
    "Party": "https://images.unsplash.com/photo-1492684223066-81342ee5ff30?w=400&auto=format&fit=crop&q=80",
    "Soulful": "https://images.unsplash.com/photo-1507838153414-b4b713384a76?w=400&auto=format&fit=crop&q=80",
}

# Additional catalogue tracks. Audio features are curated estimates for
# recommendation and discovery, not official streaming-service measurements.
EXTRA_TRACKS = [
    ("Aaromale", "ആരോമലേ", "Alphonse Puthren", "Vinnaithaandi Varuvaayaa", "Rock / Romance", "Romantic", 0.58, 0.62, 0.55, 0.42, 110.0, -6.5, 94, "5:46", "Malayalam"),
    ("Puthiyoru Pathayil", "പുതിയൊരു പാതയിൽ", "Nazriya Nazim, Sushin Shyam", "Varathan", "Indie / Alternative", "Chill", 0.54, 0.52, 0.58, 0.55, 98.0, -7.4, 92, "3:36", "Malayalam"),
    ("Theerame", "തീരമേ", "K.S. Chithra, Sooraj Santhosh", "Malik", "Melody / Soul", "Soulful", 0.42, 0.38, 0.48, 0.82, 84.0, -10.0, 93, "4:24", "Malayalam"),
    ("Parayuvaan", "പറയുവാൻ", "Sid Sriram, Neha S. Nair", "Ishq", "Acoustic / Romance", "Romantic", 0.56, 0.48, 0.68, 0.72, 92.0, -8.4, 94, "4:37", "Malayalam"),
    ("Onakka Munthiri", "ഓണക്ക മുന്തിരി", "Divya Vineeth", "Hridayam", "Pop / Feel-good", "Happy", 0.76, 0.70, 0.84, 0.28, 108.0, -6.0, 94, "3:20", "Malayalam"),
    ("Perfect", "Perfect", "Ed Sheeran", "÷ (Divide)", "Pop / Acoustic", "Romantic", 0.60, 0.45, 0.58, 0.72, 95.0, -7.0, 98, "4:23", "English"),
    ("As It Was", "As It Was", "Harry Styles", "Harry's House", "Synth-Pop", "Happy", 0.52, 0.73, 0.66, 0.34, 174.0, -5.3, 97, "2:47", "English"),
    ("Levitating", "Levitating", "Dua Lipa", "Future Nostalgia", "Disco / Pop", "Party", 0.70, 0.83, 0.92, 0.01, 103.0, -3.8, 98, "3:23", "English"),
    ("Flowers", "Flowers", "Miley Cyrus", "Endless Summer Vacation", "Pop / Dance", "Happy", 0.71, 0.68, 0.65, 0.06, 118.0, -4.8, 97, "3:20", "English"),
    ("Someone Like You", "Someone Like You", "Adele", "21", "Piano / Soul", "Heartbroken", 0.56, 0.32, 0.28, 0.89, 68.0, -8.5, 98, "4:45", "English"),
]

# 2026 entries are current chart picks, not claims that every song was released
# this year. The chart snapshot is the Global Spotify daily chart from 2026-10-15.
# Older groups contain tracks released in the named year/decade.
ERA_TRACKS = [
    # Trending on the 2026 Global Spotify daily chart
    ("Nicole Kidman", "ADÉLA", "2026 Charting Single", "Pop", "Party", 0.82, 0.86, 0.78, 0.08, 122.0, -4.2, 96, "3:10", "English", "Trending 2026"),
    ("Ain't In LA", "ADÉLA", "2026 Charting Single", "Pop / Dance", "Energetic", 0.84, 0.91, 0.80, 0.05, 128.0, -3.8, 95, "2:58", "English", "Trending 2026"),
    ("the cure", "Olivia Rodrigo", "2026 Charting Single", "Alt-Pop", "Heartbroken", 0.48, 0.52, 0.30, 0.35, 96.0, -7.0, 95, "3:24", "English", "Trending 2026"),
    ("Earrings", "Malcolm Todd", "2026 Charting Single", "Indie Pop", "Chill", 0.66, 0.56, 0.61, 0.25, 104.0, -7.2, 94, "2:42", "English", "Trending 2026"),
    ("Stop The Wedding!", "Ashe", "2026 Charting Single", "Pop / Dance", "Happy", 0.77, 0.83, 0.82, 0.10, 120.0, -4.6, 92, "3:02", "English", "Trending 2026"),
    ("stupid song", "Olivia Rodrigo", "2026 Charting Single", "Alt-Pop / Rock", "Energetic", 0.68, 0.85, 0.59, 0.08, 126.0, -4.0, 95, "3:18", "English", "Trending 2026"),
    ("Great Expectation", "SIENNA SPIRO", "2026 Charting Single", "Pop / Soul", "Soulful", 0.46, 0.45, 0.48, 0.42, 88.0, -7.8, 91, "3:35", "English", "Trending 2026"),
    ("Patient Zero", "Taylor Swift", "2026 Charting Single", "Pop", "Chill", 0.65, 0.58, 0.53, 0.18, 108.0, -6.0, 94, "3:22", "English", "Trending 2026"),
    ("Man I Need", "Olivia Dean", "2026 Charting Single", "Soul / Pop", "Romantic", 0.63, 0.55, 0.66, 0.35, 98.0, -7.1, 94, "3:04", "English", "Trending 2026"),
    ("Self Aware", "Temper City", "2026 Charting Single", "Alt-Pop", "Energetic", 0.72, 0.78, 0.62, 0.12, 116.0, -5.0, 90, "3:12", "English", "Trending 2026"),
    ("Loser", "Tame Impala", "2026 Charting Single", "Psychedelic Pop", "Chill", 0.70, 0.68, 0.58, 0.20, 110.0, -6.2, 91, "3:30", "English", "Trending 2026"),
    ("SWIM", "BTS", "2026 Charting Single", "K-Pop / Dance", "Party", 0.82, 0.90, 0.86, 0.04, 126.0, -3.7, 96, "3:14", "English", "Trending 2026"),
    ("back to friends", "sombr", "2026 Charting Single", "Indie Pop", "Heartbroken", 0.56, 0.48, 0.35, 0.38, 92.0, -7.5, 95, "3:20", "English", "Trending 2026"),
    ("Love Me Not", "Ravyn Lenae", "2026 Charting Single", "R&B / Pop", "Romantic", 0.75, 0.72, 0.70, 0.10, 116.0, -5.0, 92, "3:33", "English", "Trending 2026"),
    ("Sailor Song", "Gigi Perez", "2026 Charting Single", "Indie Folk", "Romantic", 0.46, 0.40, 0.55, 0.78, 84.0, -8.6, 93, "3:31", "English", "Trending 2026"),
    ("End of Beginning", "Djo", "2026 Charting Single", "Indie Pop", "Chill", 0.67, 0.62, 0.58, 0.16, 108.0, -6.0, 95, "2:39", "English", "Trending 2026"),

    # Songs released in 2019
    ("Old Town Road", "Lil Nas X", "7", "Country Rap", "Party", 0.88, 0.62, 0.64, 0.05, 136.0, -6.0, 97, "2:37", "English", "2019"),
    ("7 rings", "Ariana Grande", "thank u, next", "Pop / Trap", "Party", 0.78, 0.66, 0.72, 0.58, 70.0, -5.7, 96, "2:58", "English", "2019"),
    ("Dance Monkey", "Tones and I", "The Kids Are Coming", "Electropop", "Happy", 0.82, 0.59, 0.54, 0.69, 98.0, -6.0, 98, "3:29", "English", "2019"),
    ("Circles", "Post Malone", "Hollywood's Bleeding", "Pop / Soft Rock", "Chill", 0.70, 0.76, 0.55, 0.19, 120.0, -5.6, 96, "3:35", "English", "2019"),
    ("Sucker", "Jonas Brothers", "Happiness Begins", "Pop Rock", "Happy", 0.84, 0.73, 0.82, 0.04, 138.0, -5.0, 94, "3:01", "English", "2019"),
    ("Memories", "Maroon 5", "Jordi", "Pop", "Heartbroken", 0.76, 0.32, 0.55, 0.84, 91.0, -7.5, 93, "3:09", "English", "2019"),
    ("Goodbyes", "Post Malone, Young Thug", "Hollywood's Bleeding", "Hip-Hop / Pop", "Heartbroken", 0.69, 0.65, 0.33, 0.45, 150.0, -5.4, 94, "2:55", "English", "2019"),
    ("Señorita", "Shawn Mendes, Camila Cabello", "Shawn Mendes", "Latin Pop", "Romantic", 0.76, 0.54, 0.75, 0.04, 117.0, -6.0, 97, "3:11", "English", "2019"),
    ("Everything I Wanted", "Billie Eilish", "When We All Fall Asleep, Where Do We Go?", "Alt-Pop", "Sad", 0.70, 0.23, 0.24, 0.90, 120.0, -14.0, 95, "4:05", "English", "2019"),
    ("Nee Mukilo", "Vijay Yesudas, Sithara Krishnakumar", "Uyare", "Melody / Romance", "Romantic", 0.52, 0.42, 0.62, 0.78, 88.0, -8.8, 91, "4:15", "Malayalam", "2019"),
    ("Arike Naam", "Yakzan Gary Pereira, Neha Nair", "Underworld", "Pop / Romance", "Romantic", 0.61, 0.58, 0.70, 0.53, 96.0, -7.1, 88, "3:42", "Malayalam", "2019"),
    ("Varavaayi", "Shaan Rahman", "Love Action Drama", "Pop / Feel-good", "Happy", 0.68, 0.63, 0.78, 0.32, 108.0, -6.2, 88, "3:33", "Malayalam", "2019"),
    ("Pularippoo", "Sithara Krishnakumar", "Sathyam Paranja Viswasikkuvo", "Melody / Acoustic", "Chill", 0.43, 0.38, 0.60, 0.83, 82.0, -9.4, 87, "4:01", "Malayalam", "2019"),
    ("Pathinezhin Nilavil", "Vijay Yesudas, K.S. Harisankar", "Uyare", "Melody / Soul", "Soulful", 0.48, 0.42, 0.54, 0.82, 80.0, -9.2, 89, "4:02", "Malayalam", "2019"),
    ("Kaattil", "Vijay Yesudas", "Uyare", "Acoustic / Melody", "Chill", 0.40, 0.35, 0.46, 0.88, 78.0, -10.0, 86, "3:55", "Malayalam", "2019"),
    ("Kaalam Paranjath", "Shaan Rahman, Vineeth Sreenivasan", "Love Action Drama", "Pop / Upbeat", "Happy", 0.74, 0.68, 0.78, 0.28, 112.0, -5.8, 87, "3:28", "Malayalam", "2019"),
    ("Madhu Pole", "Sid Sriram, Aishwarya Ravichandran", "Dear Comrade (Malayalam)", "Melody / Romance", "Romantic", 0.58, 0.50, 0.67, 0.68, 92.0, -7.8, 90, "4:20", "Malayalam", "2019"),

    # 1990s classics
    ("Wonderwall", "Oasis", "(What's the Story) Morning Glory?", "Britpop / Rock", "Chill", 0.38, 0.89, 0.47, 0.01, 87.0, -5.0, 98, "4:18", "English", "1990s"),
    ("Creep", "Radiohead", "Pablo Honey", "Alternative Rock", "Sad", 0.52, 0.43, 0.10, 0.01, 92.0, -9.0, 95, "3:56", "English", "1990s"),
    ("I Will Always Love You", "Whitney Houston", "The Bodyguard", "Pop / Soul Ballad", "Romantic", 0.33, 0.28, 0.18, 0.91, 67.0, -12.5, 96, "4:31", "English", "1990s"),
    ("My Heart Will Go On", "Celine Dion", "Let's Talk About Love", "Pop / Film Ballad", "Romantic", 0.34, 0.40, 0.20, 0.86, 99.0, -10.0, 96, "4:40", "English", "1990s"),
    ("No Scrubs", "TLC", "FanMail", "R&B / Pop", "Party", 0.74, 0.68, 0.72, 0.03, 93.0, -5.5, 94, "3:34", "English", "1990s"),
    ("Torn", "Natalie Imbruglia", "Left of the Middle", "Pop Rock", "Heartbroken", 0.59, 0.72, 0.43, 0.02, 96.0, -6.0, 94, "4:05", "English", "1990s"),
    ("Iris", "The Goo Goo Dolls", "Dizzy Up the Girl", "Alternative Rock", "Heartbroken", 0.30, 0.72, 0.22, 0.00, 156.0, -7.0, 97, "4:49", "English", "1990s"),
    ("...Baby One More Time", "Britney Spears", "...Baby One More Time", "Dance Pop", "Happy", 0.76, 0.72, 0.79, 0.02, 93.0, -5.5, 95, "3:30", "English", "1990s"),
    ("Losing My Religion", "R.E.M.", "Out of Time", "Alternative Rock", "Soulful", 0.63, 0.86, 0.70, 0.02, 126.0, -6.0, 93, "4:28", "English", "1990s"),
    ("Pramadavanam", "K.J. Yesudas", "His Highness Abdullah", "Classical / Melody", "Soulful", 0.34, 0.30, 0.45, 0.92, 76.0, -11.0, 94, "4:35", "Malayalam", "1990s"),
    ("Devasabhathalam", "K.J. Yesudas", "His Highness Abdullah", "Classical / Devotional", "Soulful", 0.30, 0.32, 0.43, 0.94, 72.0, -11.5, 92, "4:16", "Malayalam", "1990s"),
    ("Ambalapuzhe Unnikannanodu Nee", "M.G. Sreekumar, K.S. Chithra", "Advaitham", "Melody / Classical", "Romantic", 0.42, 0.40, 0.62, 0.84, 82.0, -9.0, 93, "4:50", "Malayalam", "1990s"),
    ("Nilave Maayumo", "M.G. Sreekumar", "Minnaram", "Melody / Romance", "Romantic", 0.50, 0.44, 0.61, 0.77, 88.0, -8.5, 93, "4:20", "Malayalam", "1990s"),
    ("Vennila Chandana Kinnam", "K.J. Yesudas, Shabnam", "Azhakiya Ravanan", "Melody / Romance", "Romantic", 0.48, 0.42, 0.66, 0.80, 86.0, -8.8, 92, "4:28", "Malayalam", "1990s"),
    ("Harimuraleeravam", "M.G. Sreekumar", "Aaraam Thampuran", "Classical / Film", "Energetic", 0.56, 0.73, 0.70, 0.64, 104.0, -7.0, 94, "5:15", "Malayalam", "1990s"),
    ("Aaro Viral Meetti", "K.J. Yesudas", "Pranayavarnangal", "Melody / Soul", "Heartbroken", 0.37, 0.35, 0.28, 0.90, 76.0, -10.8, 93, "4:24", "Malayalam", "1990s"),
    ("Varamanjaladiya", "K.J. Yesudas", "Pranayavarnangal", "Melody / Romance", "Romantic", 0.46, 0.43, 0.65, 0.79, 86.0, -8.6, 92, "4:32", "Malayalam", "1990s"),
]


def generate_songs() -> pd.DataFrame:
    rows = []
    song_id = 0

    # 1. Malayalam Songs
    for item in MALAYALAM_TRACKS:
        title, title_nat, artist, album, genre, mood, dance, energy, val, acous, tempo, loud, pop, dur = item
        query = f"{title} {artist}".replace(" ", "+")
        rows.append({
            "song_id": song_id,
            "title": title,
            "title_native": title_nat,
            "artist": artist,
            "album_movie": album,
            "genre": genre,
            "language": "Malayalam",
            "mood": mood,
            "danceability": float(dance),
            "energy": float(energy),
            "valence": float(val),
            "acousticness": float(acous),
            "tempo": float(tempo),
            "loudness": float(loud),
            "popularity": int(pop),
            "duration": dur,
            "cover_url": AESTHETIC_COVERS.get(mood, AESTHETIC_COVERS["Happy"]),
            "spotify_url": f"https://open.spotify.com/search/{query}",
        })
        song_id += 1

    # 2. English Songs
    for item in ENGLISH_TRACKS:
        title, title_nat, artist, album, genre, mood, dance, energy, val, acous, tempo, loud, pop, dur = item
        query = f"{title} {artist}".replace(" ", "+")
        rows.append({
            "song_id": song_id,
            "title": title,
            "title_native": title_nat,
            "artist": artist,
            "album_movie": album,
            "genre": genre,
            "language": "English",
            "mood": mood,
            "danceability": float(dance),
            "energy": float(energy),
            "valence": float(val),
            "acousticness": float(acous),
            "tempo": float(tempo),
            "loudness": float(loud),
            "popularity": int(pop),
            "duration": dur,
            "cover_url": AESTHETIC_COVERS.get(mood, AESTHETIC_COVERS["Happy"]),
            "spotify_url": f"https://open.spotify.com/search/{query}",
        })
        song_id += 1

    for item in EXTRA_TRACKS:
        title, title_nat, artist, album, genre, mood, dance, energy, val, acous, tempo, loud, pop, dur, language = item
        query = f"{title} {artist}".replace(" ", "+")
        rows.append({
            "song_id": song_id,
            "title": title,
            "title_native": title_nat,
            "artist": artist,
            "album_movie": album,
            "genre": genre,
            "language": language,
            "mood": mood,
            "danceability": float(dance),
            "energy": float(energy),
            "valence": float(val),
            "acousticness": float(acous),
            "tempo": float(tempo),
            "loudness": float(loud),
            "popularity": int(pop),
            "duration": dur,
            "cover_url": AESTHETIC_COVERS.get(mood, AESTHETIC_COVERS["Happy"]),
            "spotify_url": f"https://open.spotify.com/search/{query}",
            "era": "",
        })
        song_id += 1

    for item in ERA_TRACKS:
        title, artist, album, genre, mood, dance, energy, val, acous, tempo, loud, pop, dur, language, era = item
        query = f"{title} {artist}".replace(" ", "+")
        rows.append({
            "song_id": song_id,
            "title": title,
            "title_native": title,
            "artist": artist,
            "album_movie": album,
            "genre": genre,
            "language": language,
            "mood": mood,
            "danceability": float(dance),
            "energy": float(energy),
            "valence": float(val),
            "acousticness": float(acous),
            "tempo": float(tempo),
            "loudness": float(loud),
            "popularity": int(pop),
            "duration": dur,
            "cover_url": AESTHETIC_COVERS.get(mood, AESTHETIC_COVERS["Happy"]),
            "spotify_url": f"https://open.spotify.com/search/{query}",
            "era": era,
        })
        song_id += 1

    return pd.DataFrame(rows)


def generate_ratings(songs: pd.DataFrame, n_users=120) -> pd.DataFrame:
    rows = []
    genres = songs["genre"].unique().tolist()

    for user_id in range(n_users):
        fav_genres = list(np.random.choice(genres, size=np.random.randint(2, 4), replace=False))
        liked_songs = songs[songs["genre"].isin(fav_genres)]
        other_songs = songs[~songs["genre"].isin(fav_genres)]

        n_liked = min(len(liked_songs), np.random.randint(10, 20))
        n_other = min(len(other_songs), np.random.randint(2, 6))

        if n_liked > 0:
            for _, song in liked_songs.sample(n_liked, random_state=user_id).iterrows():
                rating = np.clip(np.random.normal(4.5, 0.5), 2.5, 5.0)
                rows.append((user_id, int(song["song_id"]), round(float(rating), 1)))

        if n_other > 0:
            for _, song in other_songs.sample(n_other, random_state=user_id + 888).iterrows():
                rating = np.clip(np.random.normal(2.8, 0.8), 1.0, 4.0)
                rows.append((user_id, int(song["song_id"]), round(float(rating), 1)))

    return pd.DataFrame(rows, columns=["user_id", "song_id", "rating"])


DATA_DIR = Path(__file__).parent

if __name__ == "__main__":
    songs_df = generate_songs()
    ratings_df = generate_ratings(songs_df)

    songs_path = DATA_DIR / "songs.csv"
    ratings_path = DATA_DIR / "ratings.csv"

    songs_df.to_csv(songs_path, index=False, encoding="utf-8")
    ratings_df.to_csv(ratings_path, index=False, encoding="utf-8")

    print(f"Generated {len(songs_df)} curated songs ({songs_df['language'].value_counts().to_dict()})")
    print(f"Generated {len(ratings_df)} user listening ratings")
