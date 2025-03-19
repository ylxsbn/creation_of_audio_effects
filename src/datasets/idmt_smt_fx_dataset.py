import torch
import os
from xml.etree import ElementTree

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


"""https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html"""


class IDMTDataset(BaseDataset):
    """2-sec bass guitar recordings"""

    def __init__(self, effect_type, *args, **kwargs):
        # not safe
        # index_path = ROOT_PATH / "data" / "idmt-smt-dataset" / "Lists" / effect_type / "index.json"
        # if index_path.exists():
        #     index = read_json(str(index_path))
        # else:
        #     index = self._create_index(effect_type)
        index = self._create_index(effect_type)
        
        super().__init__(index, *args, **kwargs)

    def _create_index(self, effect_type):
        index = []

        dataset_path = ROOT_PATH / "data" / "idmt-smt-dataset"
        clean_metadata_path = dataset_path / "Lists" / "NoFX"
        effect_metadata_path = dataset_path / "Lists" / effect_type

        clean_input_name = {}

        for root, _, files in os.walk(clean_metadata_path):
            for file in files:
                if file.endswith(".xml"):
                    xml_path = os.path.join(root, file)

                    with open(xml_path, "r") as f:
                        tree = ElementTree.parse(f)
                    
                    xml_root = tree.getroot()

                    for audiofile in xml_root.findall("audiofile"):
                        name = audiofile.find("fileID").text

                        instrument = audiofile.find("instrument").text
                        instrumentsetting = audiofile.find("instrumentsetting").text
                        playstyle = audiofile.find("playstyle").text
                        midinr = audiofile.find("midinr").text
                        string = audiofile.find("string").text
                        fret = audiofile.find("fret").text

                        clean_id = instrument + instrumentsetting + playstyle + '-' + midinr + string + fret
                        
                        assert clean_id not in clean_input_name

                        clean_input_name[clean_id] = name

        for root, _, files in os.walk(effect_metadata_path):
            for file in files:
                if file.endswith(".xml"):
                    xml_path = os.path.join(root, file)

                    with open(xml_path, "r") as f:
                        tree = ElementTree.parse(f)
                    
                    xml_root = tree.getroot()

                    for audiofile in xml_root.findall("audiofile"):
                        fxsetting = audiofile.find("fxsetting").text

                        if int(fxsetting) != 2:
                            continue

                        name = audiofile.find("fileID").text

                        instrument = audiofile.find("instrument").text
                        instrumentsetting = audiofile.find("instrumentsetting").text
                        playstyle = audiofile.find("playstyle").text
                        midinr = audiofile.find("midinr").text
                        string = audiofile.find("string").text
                        fret = audiofile.find("fret").text

                        clean_id = instrument + instrumentsetting + playstyle + '-' + midinr + string + fret

                        index.append({"input_path": str(dataset_path / "Samples" / "NoFX" / (clean_input_name[clean_id] + ".wav")),
                                      "output_path": str(dataset_path / "Samples" / effect_type / (name + ".wav"))})


        # write index to disk
        write_json(index, str(effect_metadata_path / "index.json"))

        return index
