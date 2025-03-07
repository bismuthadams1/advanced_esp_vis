from esp_to_sdf import ESPtoSDF   


def main():
    esp_from_sdf = ESPtoSDF()
    esp_from_sdf.make_charge_sdf(sdf_file="/home/charlie-adams/advanced_esp_vis/SHP1_fixed_combined_QI_ready.sdf")

if __name__ == "__main__":
    main()