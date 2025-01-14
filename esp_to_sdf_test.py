from esp_to_sdf import ESPtoSDF   


def main():
    esp_from_sdf = ESPtoSDF()
    esp_from_sdf.make_charge_sdf(sdf_file="/Users/localadmin/Documents/molesp_test/dnmt1_ref.sdf")

if __name__ == "__main__":
    main()