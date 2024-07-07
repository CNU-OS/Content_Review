import requests

def classify_image(image_url):
    try:
        api_url = "https://api.tsinbei.com/v2/image/nsfwapi/classify"
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = response.content
            files = {'image': ('image.png', image_data)}
            response = requests.post(api_url, files=files)
            print(response.json())
            if response.status_code == 200:
                raw_data = response.json()
                rate_neutral = raw_data['neutral']
                rate_hentai = raw_data['hentai']
                rate_porn = raw_data['porn']
                rate_sexy = raw_data['sexy']
                is_safe = rate_neutral > 0.5 or rate_hentai < 0.1 or rate_porn < 0.01 or rate_sexy < 0.01
                return is_safe
            else:
                return False
        else:
            return False
    except Exception as e:
        return False