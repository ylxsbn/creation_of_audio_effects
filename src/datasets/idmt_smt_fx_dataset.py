import torch
import os
from xml.etree import ElementTree

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


# Dataset from https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html


class IDMTDataset(BaseDataset):
    """2-sec bass guitar recordings"""

    def __init__(self, part, effect_type, supervised=True, *args, **kwargs):
        self.used_setting = 2

        if not supervised:
            index = self._create_unsupervised_index(part, effect_type)
        else:
            index = self._create_index(part, effect_type)
        
        
        super().__init__(index, *args, **kwargs)

    def _create_index(self, part, effect_type):
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

                        if int(fxsetting) != self.used_setting:
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


        valid_per = 0.05
        test_per = 0.05

        valid_len = int(len(index) * valid_per)
        test_len = int(len(index) * test_per)
        train_len = len(index) - valid_len - test_len

        train, val, test = torch.utils.data.random_split(index, [train_len, valid_len, test_len],
                                                         torch.Generator().manual_seed(1337))
        
        partition = {
            "train": list(train), 
            "val": list(val),
            "test": list(test)
        }

        if not os.path.exists(effect_metadata_path / part):
            os.mkdir(effect_metadata_path / part)

        write_json(partition[part], str(effect_metadata_path / part / "index.json"))

        return partition[part]
    

    def _create_unsupervised_index(self, part, effect_type):
        index = []

        dataset_path = ROOT_PATH / "data" / "idmt-smt-dataset"
        effect_metadata_path = dataset_path / "Lists" / effect_type

        for root, _, files in os.walk(effect_metadata_path):
            for file in files:
                if file.endswith(".xml"):
                    xml_path = os.path.join(root, file)

                    with open(xml_path, "r") as f:
                        tree = ElementTree.parse(f)
                    
                    xml_root = tree.getroot()

                    for audiofile in xml_root.findall("audiofile"):
                        fxsetting = audiofile.find("fxsetting").text

                        if effect_type != "NoFX" and int(fxsetting) != self.used_setting:
                            continue

                        name = audiofile.find("fileID").text
                        index.append({"input_path": str(dataset_path / "Samples" / effect_type / (name + ".wav")),
                                      "output_path": str(dataset_path / "Samples" / effect_type / (name + ".wav"))})


        valid_per = 0.05
        test_per = 0.05

        valid_len = int(len(index) * valid_per)
        test_len = int(len(index) * test_per)
        train_len = len(index) - valid_len - test_len

        train, val, test = torch.utils.data.random_split(index, [train_len, valid_len, test_len],
                                                         torch.Generator().manual_seed(1337))
        
        partition = {
            "train": list(train), 
            "val": list(val),
            "test": list(test)
        }

        dir_name = part + "-unsupervised"
        if not os.path.exists(effect_metadata_path / dir_name):
            os.mkdir(effect_metadata_path / dir_name)

        write_json(partition[part], str(effect_metadata_path / dir_name / "index.json"))

        return partition[part]