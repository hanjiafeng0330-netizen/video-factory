"""血缘的内存实现。"""

from __future__ import annotations

from collections import defaultdict, deque

from app.domain.lineage import ArtifactRelation, TraceNode
from app.domain.refs import ArtifactRef


class InMemoryLineage:
    """同时实现 `LineageRecorder` 与 `LineageRepository`。

    合成一个类是因为写入与查询共享同一份索引；拆成两个类会立刻带来
    「两份索引如何保持一致」的问题，而这里没有任何需要拆的理由。
    """

    def __init__(self) -> None:
        self._into: dict[ArtifactRef, list[ArtifactRelation]] = defaultdict(list)
        self._out: dict[ArtifactRef, list[ArtifactRelation]] = defaultdict(list)

    def record(self, relation: ArtifactRelation) -> None:
        # 同一条关系重复记录不应产生重复行：能力重跑是常态（设计文档 3.1），
        # 而重跑不代表血缘变了。
        if relation in self._into[relation.target]:
            return
        self._into[relation.target].append(relation)
        self._out[relation.source].append(relation)

    def relations_into(self, target: ArtifactRef) -> tuple[ArtifactRelation, ...]:
        return tuple(self._into.get(target, ()))

    def relations_out_of(self, source: ArtifactRef) -> tuple[ArtifactRelation, ...]:
        return tuple(self._out.get(source, ()))

    def trace(self, ref: ArtifactRef) -> tuple[TraceNode, ...]:
        nodes: list[TraceNode] = [TraceNode(ref=ref, depth=0)]
        seen: set[ArtifactRef] = {ref}
        queue: deque[tuple[ArtifactRef, int]] = deque([(ref, 0)])

        while queue:
            current, depth = queue.popleft()
            for relation in self.relations_into(current):
                if relation.source in seen:
                    continue
                seen.add(relation.source)
                nodes.append(
                    TraceNode(
                        ref=relation.source,
                        relation_from_child=relation.relation_type,
                        depth=depth + 1,
                    )
                )
                queue.append((relation.source, depth + 1))

        return tuple(nodes)
