from esp_visualize_from_sdf import ESPFromSDF


def main():
    esp_from_sdf = ESPFromSDF()
    # esp_from_sdf.process_and_launch_esp(sdf_file="Theophylline.sdf", port=8100)
    esp_from_sdf.process_and_launch_esp(sdf_file="./SHP1_fixed_combined_QI_ready.sdf", port=5000)


if __name__ == "__main__":
    main()