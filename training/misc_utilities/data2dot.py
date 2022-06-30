# generate graphviz graphs from data transform configuration
import json
from argparse import ArgumentParser


def walk(item, parents=None, nodes=[], connections=[]):
    if isinstance(item, dict):
        assert "_tfm" in item
        tfm_name = item["_tfm"]
        this_node = len(nodes)

        if tfm_name == "Compose":
            return walk(item["transforms"], parents, nodes, connections)

        if tfm_name == "FileCache":
            attrs = {}
            for k, v in item.items():
                if k == "cache_root":
                    attrs[k] = ".../" + "/".join(v.split("/")[-3:])
                elif k != "_tfm" and k != "transforms":
                    attrs[k] = v
            nodes.append((this_node, "_FileCacheStart", attrs))
            nodes, connections, parents, _ = walk(
                item["transforms"], parents, nodes, connections
            )
            nodes.append((len(nodes), "_FileCacheEnd"))
            return nodes, connections, parents, None

        nodes.append((this_node, tfm_name))
        if parents is not None:
            for parent in parents:
                connections.append((parent, this_node, None))

        new_parents = []
        attrs = {}
        for key, value in item.items():
            if key != "_tfm":
                num_conn_before = len(connections)
                nodes, connections, parents, attr = walk(
                    value, [this_node], nodes, connections
                )
                num_conn_after = len(connections)
                if num_conn_after > num_conn_before:
                    c = connections[num_conn_before]
                    connections[num_conn_before] = (c[0], c[1], key)
                attrs[key] = attr
                if (
                    isinstance(attr, list)
                    and len(attr) == 2
                    and attr[0] is None
                    and attr[1] is None
                ):
                    print(key)
                new_parents.extend(parents)

        if tfm_name == "Rand":
            new_parents.append(this_node)

        if len(new_parents) == 0:
            new_parents = [this_node]

        n = nodes[this_node]
        nodes[this_node] = (n[0], n[1], attrs)
        return nodes, connections, list(set(new_parents)), None

    if isinstance(item, list):
        new_parents = parents
        attrs = []
        all_attrs_none = True
        for child in item:
            nodes, connections, new_parents, attr = walk(
                child, new_parents, nodes, connections
            )
            attrs.append(attr)
            if attr is not None:
                all_attrs_none = False
        if all_attrs_none:
            attrs = None
        return nodes, connections, new_parents, attrs

    return nodes, connections, [], item


def small_font(label):
    return f'<FONT POINT-SIZE="10" COLOR="#757575">{label}</FONT>'


def get_label_str(node_name, node_attrs):
    if node_attrs:
        attr_str = "<BR/>".join(
            [f"{k}: {v}" for k, v in node_attrs.items() if v is not None]
        )
        return f"<{node_name}<BR/>{small_font(attr_str)}>"
    else:
        return f'"{node_name}"'


def main(dataset_config, output_file):
    with open(dataset_config) as config_file:
        config = json.load(config_file)

    nodes = [(0, "train")]
    nodes, connections, _, _ = walk(config["train"], [0], nodes)
    nodes.append((len(nodes), "val"))
    nodes, connections, _, _ = walk(config["val"], [len(nodes) - 1], nodes, connections)

    with open(output_file, "w") as dot_file:
        dot_file.write("digraph Pipeline {\n")
        for node in nodes:
            if len(node) == 2:
                node_id, node_name = node
                node_attrs = None
            else:
                node_id, node_name, node_attrs = node
            if node_name == "_FileCacheStart":
                dot_file.write(f"subgraph cluster_{node_id} {{\n")
                dot_file.write(f'label={get_label_str("FileCache", node_attrs)}\n')
            elif node_name == "_FileCacheEnd":
                dot_file.write("}\n")
            else:
                if node_name == "train" or node_name == "val":
                    style = "shape=invtrapezium"
                else:
                    style = "style=rounded shape=box"
                dot_file.write(
                    f"{node_id} [{style} label={get_label_str(node_name, node_attrs)}]\n"
                )
        for parent, child, description in connections:
            if description is None:
                dot_file.write(f"{parent} -> {child}\n")
            else:
                dot_file.write(
                    f"{parent} -> {child} [label=<{small_font(description)}>]\n"
                )
        dot_file.write("}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "config", help="Input dataset config (usually named dataset.json)"
    )
    parser.add_argument("output", help="Output dot file")
    args = parser.parse_args()
    main(args.config, args.output)
