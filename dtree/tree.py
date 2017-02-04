#!/usr/bin/python3
import numpy as np
from node import BiNode
import networkx as nx


class CARTree(object):
    def __init__(self, maxTreeDepth=3):
        self.maxTreeDepth = maxTreeDepth
        self.Tree = nx.Graph()
        # root node
        self.Tree.add_node(0, data=BiNode(0))

    def growTree(self):
        pass

    def addNode(self):
        pass

    def getLeaves(self):
        """!
        @brief Get list of all leaf node in tree.
        """
        pass

    def _walkTree(self):
        """!
        @brief Walk down tree
        """
        pass

    def computeInputImportance(self):
        """!
        @brief Computes the normalized expnatory variable importance.
        """
        pass

    @property
    def inputImportance(self):
        pass
