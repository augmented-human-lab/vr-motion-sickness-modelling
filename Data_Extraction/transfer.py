import os
import gdown
import time

# links="https://drive.google.com/drive/folders/10R8Eehu1WWCS1uvzA_HKdcV6YFgC_Vhc?usp=sharing, https://drive.google.com/drive/folders/12oy-Ko8tk71yAv4TjTQpO9s6KK_aIQmD?usp=sharing, https://drive.google.com/drive/folders/1CHQRWGMUA85hJyfKo3F0QEcw4VnwWMzV?usp=sharing, https://drive.google.com/drive/folders/1ODhG9xnev3-iQrLMUkRuylVn836kkZqt?usp=sharing, https://drive.google.com/drive/folders/1ONZRFJGYMI3xsdF-HY36YL7nsKKmkgPn?usp=sharing, https://drive.google.com/drive/folders/1P1LWMjos30PsgrS8mexCOZ3l2SvFMSFF?usp=sharing, https://drive.google.com/drive/folders/1TUYnvJHZCyO6FNcImhhiWYgvhlsYCKJy?usp=sharing, https://drive.google.com/drive/folders/1Tkpf-6yzHYwQt0A9fzvf8TYDE1L9uWSm?usp=sharing, https://drive.google.com/drive/folders/1U-s0uIFF2ox97WIJfecZEFzyqLcwT18A?usp=sharing, https://drive.google.com/drive/folders/1ZC3JIVHVGUT_Duc3ECAk-mC_348qlqry?usp=sharing, https://drive.google.com/drive/folders/1a0Hxnl_zNlSNG93CNYtjjQ_fN8mOVd4k?usp=sharing, https://drive.google.com/drive/folders/1aAGK97LWJilQB7T2Bdd2caFB0pcSD6xD?usp=sharing, https://drive.google.com/drive/folders/1bjgyCzvXYa-mbCg6v2B895D2MJZ99Nsm?usp=sharing, https://drive.google.com/drive/folders/1cykn0H7ATHq1myV7dEVQ8bsqOO6QQ11F?usp=sharing, https://drive.google.com/drive/folders/1eZl8gWRutmQvj9LNzeUy07CEcrO7n6Kk?usp=sharing, https://drive.google.com/drive/folders/1fs8U5KeReUqnMOYLSRgnW0Dy_MGLld1a?usp=sharing, https://drive.google.com/drive/folders/1nEjGRAEkB-cXEn6OIqzQ6c-v6FyLKmjz?usp=sharing, https://drive.google.com/drive/folders/1nw-27tcERa68pDLX1_nXz8TQ8HKobWQ_?usp=sharing, https://drive.google.com/drive/folders/1qZ56rp4cbcG5LqbtHdo2bPsQgVRcZk2Y?usp=sharing, https://drive.google.com/drive/folders/1sQ2mryriITl-ILAjX7Sa_qoNipmojzzJ?usp=sharing, https://drive.google.com/drive/folders/1sTgHgnvk0swWlCy5QzxTlGpjsL34uryC?usp=sharing, https://drive.google.com/drive/folders/1utfQCw98BMFE7Z6skzSb603wukwP01iy?usp=sharing, https://drive.google.com/drive/folders/1x2PB416Kz4LBXUlSKGJYjgsTa3Eflook?usp=sharing"

# links="https://drive.google.com/drive/folders/1eFlFdyWvpwX7Gooco_A7yXSPZ5kRgwlz?usp=drive_link, https://drive.google.com/drive/folders/1l_kb5RwQcTtBn4B5Yt0PwWIDyP-TZKCr?usp=drive_link, https://drive.google.com/drive/folders/16Q0S28jFaWPgg8GS_PYUJwCbHJVP9tx3?usp=drive_link, https://drive.google.com/drive/folders/1h-nFG4qEUzp__ulpEH5evAgoiJ6qAFEW?usp=drive_link, https://drive.google.com/drive/folders/1Sf53vizKgTqKtjk81CeuFnpusICYdmCj?usp=drive_link"

links="https://drive.google.com/drive/folders/1l_kb5RwQcTtBn4B5Yt0PwWIDyP-TZKCr?usp=drive_link"

os.chdir("/data/VR_NET/zipped")
x_names=links.rsplit(", ")

for i in range(1):
    # time.sleep(3600)
    fol_id=x_names[i].rsplit("/")[5].rsplit("?")[0]
    # os.mkdir(str(i))
    url_1="https://drive.google.com/drive/folders"
    fol_path=os.path.join(url_1, fol_id)
    output='/data/VR_NET/zipped'
    print(fol_path,)
    gdown.download_folder(fol_path, quiet=False, remaining_ok=True)
    # time.sleep(3600)


