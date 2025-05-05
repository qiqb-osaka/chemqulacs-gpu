##
# @file utils.py
# @brief Utility functions for VQE calculation
# @author Yusuke Teranishi
import datetime
import json


def json_load(filename):
    with open(filename) as f:
        return json.load(f)


def json_dump(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def ndjson_dump(data, f, type="info"):
    f.write(
        json.dumps({"datetime": str(datetime.datetime.now()), "type": type, **data})
        + "\n"
    )
    f.flush()


def get_molecule(name):
    ##
    # @brief Get molecule information for environment unconnected internet
    # @param name: molecule name
    name = name.lower()
    symmetry = True
    charge = 0
    if name == "h2":
        geom = [
            ("H", (0, 0, 0)),
            ("H", (0, 0, 0.65)),
        ]
        symmetry = "D2h"
    elif name == "beh2":
        geom = [
            ("Be", (0, 0, 0)),
            ("H", (0, 0, 1.3264)),
            ("H", (0, 0, -1.3264)),
        ]
        symmetry = "D2h"
    elif name in ["h2o", "water"]:
        geom = [
            ("O", (0, 0, 0)),
            ("H", (0.2774, 0.8929, 0.2544)),
            ("H", (0.6068, -0.2383, -0.7169)),
        ]
    elif name == "n2":
        geom = [("N", (0, 0, 0.5488)), ("N", (0, 0, -0.5488))]
    elif name == "c2h4":
        geom = [
            ("C", (0.0000, 0.0000, 0.6695)),
            ("C", (0.0000, 0.0000, -0.6695)),
            ("H", (0.0000, 0.9289, 1.2321)),
            ("H", (0.0000, -0.9289, 1.2321)),
            ("H", (0.0000, 0.9289, -1.2321)),
            ("H", (0.0000, -0.9289, -1.2321)),
        ]
        symmetry = "D2h"
    elif name == "co2":
        geom = [
            ("C", (0.0000, 0.0000, 0.0000)),
            ("O", (0.0000, 0.0000, 1.1621)),
            ("O", (0.0000, 0.0000, -1.1621)),
        ]
        symmetry = "D2h"
    elif name == "c2h6":
        geom = [
            ("C", (0.0000, 0.0000, 0.7680)),
            ("C", (0.0000, 0.0000, -0.7680)),
            ("H", (-1.0192, 0.0000, 1.1573)),
            ("H", (0.5096, 0.8826, 1.1573)),
            ("H", (0.5096, -0.8826, 1.1573)),
            ("H", (1.0192, 0.0000, -1.1573)),
            ("H", (-0.5096, -0.8826, -1.1573)),
            ("H", (-0.5096, 0.8826, -1.1573)),
        ]
    elif name == "hf":
        geom = [("F", (0.0000, 0.0000, 0.0000)), ("H", (0.0000, 0.0000, 0.9168))]
        symmetry = "C2v"
    elif name == "lih":
        geom = [("Li", (0.0000, 0.0000, 0.0000)), ("H", (0.0000, 0.0000, 1.5949))]
        symmetry = "C2v"
    elif name == "nh3":
        geom = [
            ("N", (0.0000, 0.0000, 0.0000)),
            ("H", (0.0000, -0.9377, -0.3816)),
            ("H", (0.8121, 0.4689, -0.3816)),
            ("H", (-0.8121, 0.4689, -0.3816)),
        ]
        symmetry = "Cs"
    elif name == "choo-":
        geom = [
            ("O", (-1.1464, -0.1741, 0)),
            ("O", (1.1464, -0.1741, 0)),
            ("C", (0, 0.3481, 0)),
            ("H", (-0.0001, 1.4906, 0)),
        ]
        charge = -1
    elif name == "chooh":
        geom = [
            ("C", (0.0000, 0.4220, 0.0000)),
            ("O", (-1.0338, -0.4353, 0.0000)),
            ("O", (1.1587, 0.1022, 0.0000)),
            ("H", (-0.3512, 1.4612, 0.0000)),
            ("H", (-0.6482, -1.3276, 0.0000)),
        ]
        symmetry = "Cs"
    elif name == "cl2":
        geom = [
            ("Cl", (0.0000, 0.0000, 0.0000)),
            ("Cl", (0.0000, 0.0000, 1.9879)),
        ]
    elif name == "k2":
        geom = [("K", (0.0000, 0.0000, 1.7318)), ("K", (0.0000, 0.0000, -1.7318))]
        symmetry = "D2h"
    elif name == "c3h8":
        geom = [
            ("C", (0.0000000, 0.0000000, 0.5921650)),
            ("C", (0.0000000, 1.2815320, -0.2639670)),
            ("C", (0.0000000, -1.2815320, -0.2639670)),
            ("H", (0.8757870, 0.0000000, 1.2386510)),
            ("H", (-0.8757870, 0.0000000, 1.2386510)),
            ("H", (0.0000000, 2.1644520, 0.3684220)),
            ("H", (0.0000000, -2.1644520, 0.3684220)),
            ("H", (0.8795580, 1.3185820, -0.8998840)),
            ("H", (-0.8795580, 1.3185820, -0.8998840)),
            ("H", (-0.8795580, -1.3185820, -0.8998840)),
            ("H", (0.8795580, -1.3185820, -0.8998840)),
        ]
        symmetry = "C2v"
    elif name == "o3":
        geom = [
            ("O", (0.0000000, 0.0000000, 0.4601880)),
            ("O", (0.0000000, 1.0731370, -0.2300940)),
            ("O", (0.0000000, -1.0731370, -0.2300940)),
        ]
        symmetry = "C2v"
    elif name == "nacl":
        geom = [
            ("Na", (0.0000000, 0.0000000, -1.3480)),
            ("Cl", (0.0000000, 0.0000000, 0.8722)),
        ]
        symmetry = "C2v"
    elif name == "kcl":
        geom = [
            ("K", (0.0000000, 0.0000000, 1.2424)),
            ("Cl", (0.0000000, 0.0000000, -1.3885)),
        ]
        symmetry = "C2v"
    else:
        raise ValueError("Invalid molecule name")
    return geom, symmetry, charge
