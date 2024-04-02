from opentrons import protocol_api
from opentrons.types import Point
import math


metadata = {
    "protocolName": "Opentrons OT-2 pipetting performance script (P20)",
    "author": "Arthur Theuer (theuera1) <arthur.theuer@roche.com>",
    "source": "I2S & ADME Chapter, PS",
    "description": "Fill a plate with Orange G solution using the P20 multi-channel pipette to measure its pipetting performance with a plate reader.",
    "apiLevel": "2.14"
}


# Changelog V1.0
# V1.1 2023-10-13 <theuera1>: Add global variables to more easily customize the script/make it more general
# V1.2 2024-02-06 <theuera1>: Make script specific for P20 multi-channel pipette and add separate version for P300


# P20-SPECIFIC PARAMETERS:
MAX_PIPETTE_VOLUME = 20  # in µl
DYE_VOLUME = 2  # in µl (per well)
TOTAL_VOLUME = 100  # in µl (per well)
# also need to change loaded pipettes and loaded tip racks right below

# ROBOT-SPECIFIC PARAMETERS:
# Pipettes and tip racks:
LEFT_PIPETTE = "p20_single_gen2"
RIGHT_PIPETTE = "p20_multi_gen2"
TIP_RACKS = "opentrons_96_tiprack_20ul"

TESTED_PIPETTE = "right"  # use "left" or "right" to select side to test

# Deck layout:
POS_TIP_RACKS = ["1", "5"]
POS_RESERVOIR = "2"
POS_TEST_PLATE = "3"

# Additional modules:
MODULES = {"temperatureModuleV2": "9"}

# ASSAY SETTINGS:
REPLICATES = 12  # number of replicates (max 12)
WATER_VOLUME = TOTAL_VOLUME - DYE_VOLUME  # in µl (per well)

NR_OF_WATER_DISPENSES = math.ceil(WATER_VOLUME / MAX_PIPETTE_VOLUME)  # how often the pipetting is repeated per well
WATER_PER_DISPENSE = WATER_VOLUME / NR_OF_WATER_DISPENSES  # how much water is used per repeated dispense (in µl)


def run(ctx):
    # Define liquids:
    orange_g = ctx.define_liquid(name="Orange G solution", description="Orange G dissolved in deionized water (c = 1 g/l)", display_color="#F5C856")
    water = ctx.define_liquid(name="Milli-Q water", description="Deionized water for filling up the wells", display_color="#79EDFC")

    # Load modules (not needed for protocol):
    for module, pos in MODULES.items():
        ctx.load_module(module, pos)

    # Load labware:
    tipracks = []
    for pos in POS_TIP_RACKS:
        tipracks.append(ctx.load_labware(TIP_RACKS, pos))

    test_trough = ctx.load_labware("axygen_4_reservoir_70ul", POS_RESERVOIR)
    test_plate = ctx.load_labware("corning_96_wellplate_360ul_flat", POS_TEST_PLATE)

    # Load liquids:
    test_trough["A1"].load_liquid(liquid=water, volume=WATER_VOLUME * REPLICATES * 8 + 5000)
    test_trough["A2"].load_liquid(liquid=orange_g, volume=DYE_VOLUME * REPLICATES * 8 + 5000)

    # Load pipettes:
    left = ctx.load_instrument(LEFT_PIPETTE, "left", tip_racks=tipracks)
    right = ctx.load_instrument(RIGHT_PIPETTE, "right", tip_racks=tipracks)

    if TESTED_PIPETTE == "right":
        pipette = right
    elif TESTED_PIPETTE == "left":
        pipette = left
    else:
        pipette = None

    # Reset the tip box so that all tips are available:
    pipette.reset_tipracks()

    # Water pipetting loop:
    for i in range(REPLICATES):
        pipette.pick_up_tip()
        for _ in range(NR_OF_WATER_DISPENSES):
            pipette.aspirate(WATER_PER_DISPENSE, test_trough["A1"])

            well_pos = test_plate.columns()[i][0]
            pipette.dispense(WATER_PER_DISPENSE, well_pos)
            pipette.move_to(well_pos.top(), speed=5)

        pipette.return_tip()

    # Dye pipetting loop:
    for i in range(REPLICATES):
        pipette.pick_up_tip()
        pipette.aspirate(DYE_VOLUME, test_trough["A2"])

        well_pos = test_plate.columns()[i][0]
        pipette.dispense(DYE_VOLUME, well_pos)
        pipette.move_to(well_pos.top(), speed=5)

        pipette.return_tip()
