from google_images_download import google_images_download

# 定义运动项目列表
sports_list = [
    'Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics', 'Swimming',
    'Badminton', 'Sailing', 'Gymnastics', 'Art Competitions', 'Handball',
    'Weightlifting', 'Wrestling', 'Water Polo', 'Hockey', 'Rowing', 'Fencing',
    'Equestrianism', 'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving',
    'Canoeing', 'Tennis', 'Modern Pentathlon', 'Golf', 'Softball', 'Archery',
    'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
    'Rhythmic Gymnastics', 'Rugby Sevens', 'Trampolining', 'Beach Volleyball',
    'Triathlon', 'Rugby', 'Lacrosse', 'Polo', 'Cricket', 'Ice Hockey', 'Racquets',
    'Motorboating', 'Croquet', 'Figure Skating', 'Jeu De Paume', 'Roque',
    'Basque Pelota', 'Alpinism', 'Aeronautics', 'Cycling Road',
    'Artistic Gymnastics', 'Karate', 'Baseball/Softball',
    'Trampoline Gymnastics', 'Marathon Swimming', 'Canoe Slalom', 'RUrfing',
    'Canoe Sprint', 'Cycling BMX Racing', 'Equestrian', 'Artistic Swimming',
    'Cycling Track', 'Skateboarding', 'Cycling Mountain Bike', '3x3 Basketball',
    'Cycling BMX Freestyle', 'Sport Climbing', 'Marathon Swimming, Swimming',
    'Breaking', 'Cycling Road, Cycling Track',
    'Cycling Road, Cycling Mountain Bike', 'Cycling Road, Triathlon',
    '3x3 Basketball, Basketball'
]

# 初始化图片下载器
response = google_images_download.googleimagesdownload()

# 下载图片函数
def download_images_for_sports(sports):
    for sport in sports:
        try:
            print(f"Downloading images for: {sport}")
            arguments = {
                "keywords": sport,
                "limit": 1,  # 每个项目下载1张图片
                "print_urls": True,  # 打印图片URL
                "output_directory": f"./photo",  # 输出目录
                "image_directory": sport.replace(" ", "_"),  # 每个项目单独文件夹
                "format": "jpg"  # 下载图片格式
            }
            response.download(arguments)
        except Exception as e:
            print(f"Error downloading images for {sport}: {e}")

# 执行图片下载
download_images_for_sports(sports_list)
