from merklelib import MerkleTree
import hashlib
from enum import Enum

PROTOCOL_VERSION = 0.1


class ReproducibilityFlags(Enum):
    """
    Enum for supported reproducibility modes.
    TODO: Link to more detailed description
    """
    NOTHING = 0
    RERUN = 1
    REPEAT = 2
    RECOMPUTE = 4
    REPRODUCE = 5
    REPLICATE_SCI = 6  # Rerun + Reproduce
    REPLICATE_COMP = 7  # Recompute + Reproduce
    REPLICATE_TOTAL = 8  # Repeat + Reproduce
    EXPERIMENTAL = 9


REPRO_DEFAULT = ReproducibilityFlags.NOTHING
HASHING_ALG = hashlib.sha3_256


def rflag_caster(val, default=REPRO_DEFAULT):
    """
    Function to safely cast strings and ints to their appropriate ReproducibilityFlag
    E.g. rflag_caster(1) -> ReproducibilityFlag.RERUN
    E.g. rlag_caster("3") -> ReproducibilityFlag.REPRODUCE
    E.g. rflag_caster("two") -> REPRO_DEFAULT
    :param val: The passed value (either int or str)
    :param default: The default value to be returned upon failure
    :return: Appropriate ReproducibilityFlag
    """
    if type(val) == str:
        try:
            return ReproducibilityFlags(int(val))
        except(ValueError, TypeError):
            return default
    elif type(val) == int:
        try:
            return ReproducibilityFlags(val)
        except(ValueError, TypeError):
            return default
    elif type(val) is None:
        return default


def rmode_supported(flag: ReproducibilityFlags):
    """
    Determines in a given flag is currently supported.
    A slightly pedantic solution but it does centralize the process.
    There is the possibility that different functionality is possible on a per-install basis.
    Named to be used as a if rmode_supported(flag)
    :param flag: A ReproducibilityFlag enum being queried
    :return: True if supported, False otherwise
    """
    if type(flag) != ReproducibilityFlags:
        raise TypeError("Need to be working with a ReproducibilityFlag enum")
    if flag == ReproducibilityFlags.NOTHING \
            or flag == ReproducibilityFlags.RERUN \
            or flag == ReproducibilityFlags.REPEAT \
            or flag == ReproducibilityFlags.RECOMPUTE \
            or flag == ReproducibilityFlags.REPRODUCE \
            or flag == ReproducibilityFlags.REPLICATE_SCI \
            or flag == ReproducibilityFlags.REPLICATE_COMP \
            or flag == ReproducibilityFlags.REPLICATE_TOTAL \
            or flag == ReproducibilityFlags.EXPERIMENTAL:
        return True
    else:
        return False


def common_hash(value):
    return HASHING_ALG(value).hexdigest()


def generate_memory_reprodata(data, inports, outports):
    rout = {'lgt_data': {
        'category_type': "Data",
        'category': "Memory",
        'numInputPorts': inports,
        'numOutputPorts': outports,
        'streaming': False,
    }, 'lg_data': {
        'data_volume': '5',
    }, 'pgt_data': {
        'type': 'plain',
        'storage': 'Memory',
        'rank': [0],
        'node': "0",
        'island': "0"

    }, 'pg_data': {
        'node': '127.0.0.1',
        'island': '127.0.0.1'
    }, 'rg_data': {
        'status': 2,
        'data_hash': common_hash(data)
    }}
    return rout


def generate_file_reprodata(data, filename, inports, outports):
    rout = {'lgt_data': {
        'category_type': "Data",
        'category': "File",
        'numInputPorts': inports,
        'numOutputPorts': outports,
        'streaming': False,
    }, 'lg_data': {
        'data_volume': '5',
        'check_filepath_exists': '0',
        'filepath': filename,
        'dirname': '',
    }, 'pgt_data': {
        'type': 'plain',
        'storage': 'File',
        'rank': [0],
        'node': "0",
        'island': "0"

    }, 'pg_data': {
        'node': '127.0.0.1',
        'island': '127.0.0.1'
    }, 'rg_data': {
        'status': 2,
        'data_hash': common_hash(data)
    }}
    return rout


def filter_component_reprodata(component: dict, rmode: ReproducibilityFlags):
    # LGT_Merkleroot
    if rmode == ReproducibilityFlags.REPRODUCE:
        component['lgt_data'].pop('numInputPorts')
        component['lgt_data'].pop('numOutputPorts')
        component['lgt_data'].pop('streaming')
    lgt_root = MerkleTree(component['lgt_data'].items(), common_hash).merkle_root
    component['lgt_data']['merkleroot'] = lgt_root

    # LG_Merkleroot
    if not (rmode == ReproducibilityFlags.RECOMPUTE or rmode == ReproducibilityFlags.REPLICATE_COMP or rmode == ReproducibilityFlags.REPLICATE_TOTAL):
        # Remove filenames and dirnames
        if component['lgt_data']['category'] == 'File':
            component['lg_data'].pop('filepath')
            component['lg_data'].pop('dirname')
    if (rmode == ReproducibilityFlags.RERUN or rmode == ReproducibilityFlags.REPRODUCE or rmode == ReproducibilityFlags.REPLICATE_SCI):
        # Remove everything
        component['lg_data'] = {}

    lg_root = MerkleTree(component['lg_data'].items(), common_hash).merkle_root
    component['lg_data']['merkleroot'] = lg_root

    # PGT_Merkleroot
    """
    if rmode != ReproducibilityFlags.REPRODUCE:
        if component['pgt_data']['type'] == 'plain':
            component['pgt_data'].pop('storage')
    """
    if rmode == ReproducibilityFlags.REPRODUCE and component['pgt_data']['type'] == 'app':
        component['pgt_data'].pop('dt')
    if rmode != ReproducibilityFlags.REPLICATE_COMP or rmode != ReproducibilityFlags.RECOMPUTE:
        component['pgt_data'].pop('rank')
        component['pgt_data'].pop('node')
        component['pgt_data'].pop('island')
    pgt_root = MerkleTree(component['pgt_data'].items(), common_hash).merkle_root
    component['pgt_data']['merkleroot'] = pgt_root

    # PG_Merkleroot
    if rmode != ReproducibilityFlags.REPLICATE_COMP or rmode != ReproducibilityFlags.RECOMPUTE:
        component['pg_data'] = {}
    pg_root = MerkleTree(component['pg_data'].items(), common_hash).merkle_root
    component['pg_data']['merkleroot'] = pg_root

    # RG_Merkleroot
    if component['pgt_data']['type'] == 'plain' and rmode.value <= ReproducibilityFlags.RECOMPUTE.value:
        component['rg_data'].pop('data_hash')
    if component['pgt_data']['type'] == 'app' and not (rmode.value == ReproducibilityFlags.RECOMPUTE.value or rmode.value == ReproducibilityFlags.REPLICATE_COMP.value):
        component['rg_data'] = {'status': component['rg_data']['status']}
    if rmode.value == ReproducibilityFlags.REPRODUCE.value:
        component['rg_data'].pop('status')
    rg_root = MerkleTree(component['rg_data'].items(), common_hash).merkle_root
    component['rg_data']['merkleroot'] = rg_root
    return component


def build_block(component, parents, block_data, abstraction: str):
    parenthashes = abstraction + "_parenthashes"
    blockhashes = abstraction + "_blockhash"
    data = abstraction + "_data"
    component[parenthashes] = {}
    i = 0
    for parent in parents:
        component[parenthashes].update({i: parent[blockhashes]})
        i += 1
    if abstraction == 'lg' and 'merkleroot' in component[data]:
        hash = component[data]['merkleroot']
        block_data.append(hash)
    for parenthash in sorted(component[parenthashes].values()):
        block_data.append(parenthash)
    mtree = MerkleTree(block_data, common_hash)
    component[blockhashes] = mtree.merkle_root


def chain_parents(component: dict, parents: list):
    # LG(T)
    block_data = [component['lgt_data']['merkleroot']]
    build_block(component, parents, block_data, 'lg')
    # PGT
    block_data = [component['pgt_data']['merkleroot'],
                  component['lg_blockhash']]
    build_block(component, parents, block_data, 'pgt')
    # PG
    block_data = [component['pg_data']['merkleroot'],
                  component['pgt_blockhash'],
                  component['lg_blockhash']]
    build_block(component, parents, block_data, 'pg')
    # RG
    block_data = [component['rg_data']['merkleroot'],
                  component['pg_blockhash'],
                  component['pgt_blockhash'],
                  component['lg_blockhash']]
    build_block(component, parents, block_data, 'rg')


def generate_reprodata(rmode: ReproducibilityFlags):
    reprodata = {'rmode': str(rmode.value), 'meta_data': {
        'repro_protocol': 0.1,
        'hashing_alg': "_sha3.sha3_256"
    }}
    merke_tree = MerkleTree(reprodata.items(), common_hash)
    reprodata['merkleroot'] = merke_tree.merkle_root
    return reprodata


def agglomerate_leaves(leaves: list):
    merkletree = MerkleTree(sorted(leaves))
    return merkletree.merkle_root
