from esp_visualize_from_sdf import ESPFromSDF


def main():
    esp_from_sdf = ESPFromSDF()
    # esp_from_sdf.process_and_launch_esp(sdf_file="Theophylline.sdf", port=8100)
    esp_from_sdf.process_and_launch_esp(sdf_file="./shp2_bond_changed_QI_read_ligand.pdb", pdb=True, port=7000)


if __name__ == "__main__":
    main()