import random
import string
import sys

from pymilvus import MilvusClient, DataType
from multiprocessing import Process


def insert_data(start):
    data = []
    for i in range(start, 2000+start):
        data.append({
            "id": i,
            "vector": [random.uniform(-1, 1) for _ in range(768)]
            # , "varchar_1": rand_string()
        })
    return data

#  CLUSTER_ENDPOINT sys.argv[1]
#  token sys.argv[2]
#  start sys.argv[3]
#  batch sys.argv[4]
def func_test(coll_name):
    # CLUSTER_ENDPOINT = "https://in01-b80bbc7748deadd.aws-us-west-2.vectordb-uat3.zillizcloud.com:19544"
    # CLUSTER_ENDPOINT = "https://in01-26de4d26fdfeac6.aws-us-west-2.vectordb-uat3.zillizcloud.com:19534"
    CLUSTER_ENDPOINT = sys.argv[1]
    token = sys.argv[2]
    # 1. Set up a Milvus client
    client = MilvusClient(
        uri=CLUSTER_ENDPOINT,
        token=token
    )

    collection_name = coll_name

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
        # highlight-next-line
        # partition_key_field="varchar_1",
        num_partitions=1024
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
    # field_name = "varchar_1"
    # schema.add_field(field_name=field_name, datatype=DataType.VARCHAR, max_length=5000)

    index_params = MilvusClient.prepare_index_params()

    index_params.add_index(
        field_name="id",
        index_type="STL_SORT"
    )

    # field_name = f"varchar_1"
    # index_params.add_index(
    #     field_name=field_name,
    #     index_type="Trie"
    # )

    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="L2",
        params={}
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        # update shard number
        shards_num=16
    )

    # colors = ["green", "blue", "yellow", "red", "black", "white", "purple", "pink", "orange", "brown", "grey"]

    num_entities = 10000
    for i in range(num_entities//2000):
        data = insert_data(i*2000)
        res = client.insert(
            collection_name=collection_name,
            data=data
        )
        # print(res)

    query_vectors = [[random.uniform(-1, 1) for _ in range(768)]]

    res = client.search(
        collection_name=collection_name,
        data=query_vectors,
        # filter="varchar_1 in  [\"a\", \"aa\", \"aaa\", \"aaaa\", \"aaaaa\" , \"aaaaaa\" ] ",
        search_params={"metric_type": "L2", "params": {"search_list": 32}},
        output_fields=["id"],
        limit=3
    )
    print("res: ", end=" ")
    print(res)
    # print(client.query(expr="", output_fields=["count(*)"]))


if __name__ == '__main__':
        sta = int(sys.argv[3])
        batch = int(sys.argv[4])
        rand = bool(sys.argv[5])
        concurrent = bool(sys.argv[6])
        process_list = []
            
        if not concurrent:
            for i in range(sta, sta + batch):
                if rand:
                    coll_name = "coll_" + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(4))
                else:
                    coll_name = "coll_" + str(i)
                func_test(coll_name)
        else:
            processes = []
            for i in range(sta, sta + batch):
                if rand:
                    coll_name = "coll_" + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(4))
                else:
                    coll_name = "coll_" + str(i)
                p = Process(target=func_test, args=(coll_name,))
                p.start()
                processes.append(p)
            # for p in processes:
            #     p.join()
