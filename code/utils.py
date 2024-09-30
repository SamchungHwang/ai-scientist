import os
from PIL import Image
from io import BytesIO


def save_graph2(graph, image_path):
    # 그래프 이미지를 파일로 저장하는 코드
    try:
        # Mermaid PNG 데이터를 메모리에 저장합니다.
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        # BytesIO를 사용하여 메모리의 데이터를 읽고 Pillow를 통해 이미지를 저장합니다.
        with open(image_path, "wb") as f:
            f.write(png_data)

        # 저장된 이미지 파일이 존재하는지 확인하고 시스템에서 엽니다.
        if os.path.exists(image_path):
            print(f"Graph image saved at {image_path}.")
            # 시스템의 기본 이미지 뷰어로 이미지를 엽니다 (Windows, Mac, Linux 모두 지원).
            os.system(
                f"open {image_path}" if os.name == "posix" else f"start {image_path}"
            )
        else:
            print("Graph image could not be saved.")
    except Exception as e:
        print(f"An error occurred: {e}")


def save_graph(graph, image_path):
    from graphviz import Digraph

    dot = Digraph()

    # Add nodes
    for node in graph.nodes:
        dot.node(node)

    # Add edges
    for edge in graph.edges:
        dot.edge(edge[0], edge[1])

    dot.render(image_path, format="png")
