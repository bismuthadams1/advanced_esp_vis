from esp_to_sdf import ESPtoSDF   


def main():
    esp_from_sdf = ESPtoSDF()
    esp_from_sdf.make_charge_sdf(sdf_file="/Users/localadmin/Documents/Documents - CMDADMIN’s MacBook Air/molesp_test/ku.a.003.02_ph4.sdf")

if __name__ == "__main__":
    main()