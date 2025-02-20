from esp_visualize_from_sdf import ESPFromSDF


def main():
    esp_from_sdf = ESPFromSDF()
    # esp_from_sdf.process_and_launch_esp(sdf_file="Theophylline.sdf", port=8100)
    esp_from_sdf.process_and_launch_esp(sdf_file="/Users/localadmin/Documents/Documents - CMDADMIN’s MacBook Air/projects/esp_complementarity/Experiment_balanol/ligand.sdf", port=8200)


if __name__ == "__main__":
    main()