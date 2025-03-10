from esp_visualize_from_sdf import ESPFromSDF


def main():
    esp_from_sdf = ESPFromSDF()
    # esp_from_sdf.process_and_launch_esp(sdf_file="Theophylline.sdf", port=8100)
    esp_from_sdf.process_and_launch_esp(
        sdf_file="/home/charlie-adams/advanced_esp_vis/shp2_bond_changed_QI_ready.pdb",
        pdb=True,
        port=7000,
        charge_model = 'MBIS_WB_GAS_CHARGE',
    )


if __name__ == "__main__":
    main()