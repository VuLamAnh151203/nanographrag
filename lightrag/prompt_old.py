from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to user query about Knowledge Base provided below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base."""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Conversation History:
{history}

Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

# PROMPTS["keywords_extraction_examples"] = [
#     """Example 1:

# Query: "How does international trade influence global economic stability?"
# ################
# Output:
# {
#   "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
#   "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
# }
# #############################""",
#     """Example 2:

# Query: "What are the environmental consequences of deforestation on biodiversity?"
# ################
# Output:
# {
#   "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
#   "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
# }
# #############################""",
#     """Example 3:

# Query: "What is the role of education in reducing poverty?"
# ################
# Output:
# {
#   "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
#   "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
# }
# #############################""",
# ]

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{
  "high_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"],
  "low_level_keywords": ["International trade", "Global economic stability", "Economic impact"]
}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{
  "high_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"],
  "low_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"]
}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{
  "high_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"],
  "low_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"]
}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks."""


PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate whether these two questions are semantically similar, and whether the answer to Question 2 can be used to answer Question 1, provide a similarity score between 0 and 1 directly.

Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["mix_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

2. From Document Chunks(DC):
{vector_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sesctions focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" sesction. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), in the following format: [KG/DC] Source content
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""




# Thêm các prompt tiếng Việt
PROMPTS["DEFAULT_LANGUAGE_VI"] = "Tiếng Việt"
PROMPTS["DEFAULT_TUPLE_DELIMITER_VI"] = PROMPTS["DEFAULT_TUPLE_DELIMITER"]  # Giữ nguyên
PROMPTS["DEFAULT_RECORD_DELIMITER_VI"] = PROMPTS["DEFAULT_RECORD_DELIMITER"]  # Giữ nguyên
PROMPTS["DEFAULT_COMPLETION_DELIMITER_VI"] = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]  # Giữ nguyên
PROMPTS["process_tickers_VI"] = PROMPTS["process_tickers"]  # Giữ nguyên

PROMPTS["DEFAULT_ENTITY_TYPES_VI"] = ["tổ chức", "người", "địa điểm", "sự kiện", "danh mục"]

PROMPTS["entity_extraction_VI"] = """-Mục tiêu-
Khi nhận được một văn bản có thể liên quan đến hoạt động này và một danh sách các loại thực thể, hãy xác định tất cả các thực thể thuộc các loại đó từ văn bản và tất cả các mối quan hệ giữa các thực thể đã xác định.
Sử dụng {language} làm ngôn ngữ đầu ra.

-Các bước-
1. Xác định tất cả các thực thể. Đối với mỗi thực thể được xác định, trích xuất thông tin sau:
- entity_name: Tên của thực thể, sử dụng cùng ngôn ngữ như văn bản đầu vào. Nếu là tiếng Việt, viết hoa tên.
- entity_type: Một trong các loại sau: [{entity_types}]
- entity_description: Mô tả toàn diện về thuộc tính và hoạt động của thực thể
Định dạng mỗi thực thể như sau ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Từ các thực thể được xác định ở bước 1, xác định tất cả các cặp (source_entity, target_entity) có *mối quan hệ rõ ràng* với nhau.
Đối với mỗi cặp thực thể có liên quan, trích xuất thông tin sau:
- source_entity: tên của thực thể nguồn, như được xác định ở bước 1
- target_entity: tên của thực thể đích, như được xác định ở bước 1
- relationship_description: giải thích lý do bạn nghĩ thực thể nguồn và thực thể đích có liên quan với nhau
- relationship_strength: điểm số chỉ độ mạnh của mối quan hệ giữa thực thể nguồn và thực thể đích
- relationship_keywords: một hoặc nhiều từ khóa quan trọng tóm tắt bản chất tổng thể của mối quan hệ, tập trung vào các khái niệm hoặc chủ đề hơn là các chi tiết cụ thể
Định dạng mỗi mối quan hệ như sau ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Xác định các từ khóa quan trọng tóm tắt các khái niệm, chủ đề hoặc chủ đề chính của toàn bộ văn bản. Những từ khóa này phải nắm bắt được các ý tưởng tổng thể có trong tài liệu.
Định dạng các từ khóa cấp nội dung như sau ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Trả về đầu ra bằng {language} dưới dạng một danh sách đơn lẻ tất cả các thực thể và mối quan hệ được xác định trong bước 1 và 2. Sử dụng **{record_delimiter}** làm dấu phân cách danh sách.

5. Khi hoàn thành, hãy đưa ra {completion_delimiter}

######################
-Ví dụ-
######################
{examples}

#############################
-Dữ liệu thực-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples_VI"] = [
    """Ví dụ 1:

Entity_types: [người, công nghệ, nhiệm vụ, tổ chức, vị trí]
Text:
trong khi Alex nghiến chặt hàm, cơn bức xúc âm ỉ trước thái độ chắc nịch của Taylor. Chính sự cạnh tranh ngầm này khiến anh luôn tỉnh táo, cảm giác rằng cam kết chung của anh và Jordan với khám phá là một sự nổi loạn thầm lặng chống lại tầm nhìn thu hẹp về kiểm soát và trật tự của Cruz.

Rồi Taylor làm điều gì đó bất ngờ. Họ dừng lại bên cạnh Jordan và, trong giây lát, quan sát thiết bị với điều gì đó gần như sùng kính. "Nếu công nghệ này có thể được hiểu..." Taylor nói, giọng họ nhỏ hơn, "Nó có thể thay đổi cuộc chơi cho chúng ta. Cho tất cả chúng ta."

Sự bác bỏ ngầm trước đó dường như chùng xuống, thay thế bằng một thoáng tôn trọng miễn cưỡng cho tầm quan trọng của những gì nằm trong tay họ. Jordan ngước lên, và trong một nhịp tim thoáng qua, mắt họ khóa với Taylor, một cuộc đối đầu không lời về ý chí dịu xuống thành một lệnh đình chiến không ổn định.

Đó là một sự biến đổi nhỏ, khó nhận thấy, nhưng Alex đã ghi nhận với một cái gật đầu thầm lặng. Tất cả họ đã được đưa đến đây bởi những con đường khác nhau
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"người"{tuple_delimiter}"Alex là một nhân vật cảm thấy bực tức và quan sát nhạy bén về mối quan hệ giữa các nhân vật khác."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"người"{tuple_delimiter}"Taylor được mô tả với sự chắc nịch và thể hiện khoảnh khắc tôn kính đối với một thiết bị, cho thấy sự thay đổi trong quan điểm."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"người"{tuple_delimiter}"Jordan chia sẻ cam kết khám phá và có tương tác quan trọng với Taylor liên quan đến một thiết bị."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"người"{tuple_delimiter}"Cruz được liên kết với tầm nhìn về kiểm soát và trật tự, ảnh hưởng đến mối quan hệ giữa các nhân vật khác."){record_delimiter}
("entity"{tuple_delimiter}"Thiết Bị"{tuple_delimiter}"công nghệ"{tuple_delimiter}"Thiết Bị đóng vai trò trung tâm trong câu chuyện, với tiềm năng thay đổi cuộc chơi, và được Taylor tôn kính."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex bị ảnh hưởng bởi thái độ chắc nịch của Taylor và quan sát sự thay đổi trong thái độ của Taylor đối với thiết bị."{tuple_delimiter}"động lực quyền lực, sự thay đổi quan điểm"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex và Jordan chia sẻ cam kết với khám phá, điều này trái ngược với tầm nhìn của Cruz."{tuple_delimiter}"mục tiêu chung, nổi loạn"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor và Jordan tương tác trực tiếp liên quan đến thiết bị, dẫn đến khoảnh khắc tôn trọng lẫn nhau và một lệnh đình chiến không ổn định."{tuple_delimiter}"giải quyết xung đột, tôn trọng lẫn nhau"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Cam kết của Jordan với khám phá là một sự nổi loạn chống lại tầm nhìn kiểm soát và trật tự của Cruz."{tuple_delimiter}"xung đột tư tưởng, nổi loạn"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Thiết Bị"{tuple_delimiter}"Taylor thể hiện sự tôn kính đối với thiết bị, cho thấy tầm quan trọng và tiềm năng tác động của nó."{tuple_delimiter}"tôn kính, ý nghĩa công nghệ"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"động lực quyền lực, xung đột tư tưởng, khám phá, nổi loạn"){completion_delimiter}
#############################""",
    """Ví dụ 2:

Entity_types: [người, công nghệ, nhiệm vụ, tổ chức, vị trí]
Text:
Họ không còn là những nhân viên đơn thuần; họ đã trở thành những người gác cổng của một ngưỡng cửa, những người gìn giữ một thông điệp từ một cõi bên kia những vì sao và những lá cờ. Sự nâng cao này trong nhiệm vụ của họ không thể bị ràng buộc bởi các quy định và giao thức đã thiết lập—nó đòi hỏi một quan điểm mới, một quyết tâm mới.

Căng thẳng xuyên suốt cuộc đối thoại của những tiếng bíp và tĩnh điện khi thông tin liên lạc với Washington vang lên trong nền. Nhóm đứng, một bầu không khí điềm báo bao trùm họ. Rõ ràng rằng những quyết định họ đưa ra trong những giờ tới có thể tái định nghĩa vị trí của nhân loại trong vũ trụ hoặc kết án họ vào sự thiếu hiểu biết và nguy hiểm tiềm tàng.

Kết nối của họ với các vì sao đã được củng cố, nhóm chuyển sang giải quyết cảnh báo đang hình thành, chuyển từ người nhận thụ động sang người tham gia chủ động. Bản năng sau của Mercer đã trở nên quan trọng hơn—nhiệm vụ của nhóm đã tiến hóa, không còn chỉ để quan sát và báo cáo mà là để tương tác và chuẩn bị. Một sự biến đổi đã bắt đầu, và Chiến dịch: Dulce vang lên với tần số mới của sự dũng cảm của họ, một âm điệu không phải do
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"vị trí"{tuple_delimiter}"Washington là một địa điểm nơi thông tin liên lạc đang được nhận, cho thấy tầm quan trọng của nó trong quá trình ra quyết định."){record_delimiter}
("entity"{tuple_delimiter}"Chiến dịch: Dulce"{tuple_delimiter}"nhiệm vụ"{tuple_delimiter}"Chiến dịch: Dulce được mô tả là một nhiệm vụ đã phát triển để tương tác và chuẩn bị, cho thấy sự thay đổi đáng kể trong mục tiêu và hoạt động."){record_delimiter}
("entity"{tuple_delimiter}"Nhóm"{tuple_delimiter}"tổ chức"{tuple_delimiter}"Nhóm được mô tả là một nhóm các cá nhân đã chuyển từ người quan sát thụ động sang người tham gia tích cực trong một nhiệm vụ, thể hiện sự thay đổi năng động trong vai trò của họ."){record_delimiter}
("entity"{tuple_delimiter}"Mercer"{tuple_delimiter}"người"{tuple_delimiter}"Mercer là người có bản năng đã trở nên quan trọng hơn trong bối cảnh nhiệm vụ đang phát triển."){record_delimiter}
("relationship"{tuple_delimiter}"Nhóm"{tuple_delimiter}"Washington"{tuple_delimiter}"Nhóm nhận thông tin liên lạc từ Washington, điều này ảnh hưởng đến quá trình ra quyết định của họ."{tuple_delimiter}"ra quyết định, ảnh hưởng bên ngoài"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Nhóm"{tuple_delimiter}"Chiến dịch: Dulce"{tuple_delimiter}"Nhóm tham gia trực tiếp vào Chiến dịch: Dulce, thực hiện các mục tiêu và hoạt động đã phát triển của nó."{tuple_delimiter}"phát triển nhiệm vụ, tham gia tích cực"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Mercer"{tuple_delimiter}"Nhóm"{tuple_delimiter}"Bản năng của Mercer đã có ảnh hưởng đến sự phát triển nhiệm vụ của nhóm từ quan sát thụ động sang tương tác tích cực."{tuple_delimiter}"ảnh hưởng, phát triển chiến lược"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"phát triển nhiệm vụ, ra quyết định, tham gia tích cực, ý nghĩa vũ trụ"){completion_delimiter}
#############################""",
    """Ví dụ 3:

Entity_types: [người, vai trò, công nghệ, tổ chức, sự kiện, vị trí, khái niệm]
Text:
giọng họ cắt qua tiếng ồn của hoạt động. "Kiểm soát có thể là một ảo tưởng khi đối mặt với một trí thông minh theo đúng nghĩa đen viết ra luật lệ của chính nó," họ tuyên bố một cách điềm tĩnh, ném một cái nhìn cảnh giác lên dòng dữ liệu dồn dập.

"Giống như nó đang học cách giao tiếp," Sam Rivera đề nghị từ một giao diện gần đó, sự năng động trẻ trung của họ báo trước một sự pha trộn giữa kinh ngạc và lo lắng. "Điều này mang đến một ý nghĩa hoàn toàn mới cho 'nói chuyện với người lạ'."

Alex khảo sát đội của mình—mỗi khuôn mặt đều thể hiện sự tập trung, quyết tâm, và không ít sự lo lắng. "Đây rất có thể là liên lạc đầu tiên của chúng ta," anh thừa nhận, "Và chúng ta cần sẵn sàng cho bất cứ điều gì trả lời lại."

Cùng nhau, họ đứng trên bờ vực của điều chưa biết, đúc kết phản ứng của nhân loại đối với một thông điệp từ bầu trời. Sự im lặng tiếp theo là hữu hình—một sự nội quan tập thể về vai trò của họ trong vở kịch vũ trụ vĩ đại này, một vở kịch có thể viết lại lịch sử loài người.

Cuộc đối thoại được mã hóa tiếp tục diễn ra, các mẫu phức tạp của nó cho thấy một sự dự đoán gần như bí ẩn
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"người"{tuple_delimiter}"Sam Rivera là thành viên của một đội đang làm việc để giao tiếp với một trí thông minh không xác định, thể hiện sự pha trộn giữa kinh ngạc và lo lắng."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"người"{tuple_delimiter}"Alex là người lãnh đạo của một đội đang cố gắng thiết lập liên lạc đầu tiên với một trí thông minh không xác định, thừa nhận tầm quan trọng của nhiệm vụ của họ."){record_delimiter}
("entity"{tuple_delimiter}"Kiểm soát"{tuple_delimiter}"khái niệm"{tuple_delimiter}"Kiểm soát đề cập đến khả năng quản lý hoặc điều hành, điều bị thách thức bởi một trí thông minh tự viết ra luật lệ của chính mình."){record_delimiter}
("entity"{tuple_delimiter}"Trí thông minh"{tuple_delimiter}"khái niệm"{tuple_delimiter}"Trí thông minh ở đây đề cập đến một thực thể không xác định có khả năng viết ra luật lệ của chính mình và học cách giao tiếp."){record_delimiter}
("entity"{tuple_delimiter}"Liên lạc đầu tiên"{tuple_delimiter}"sự kiện"{tuple_delimiter}"Liên lạc đầu tiên là khả năng giao tiếp ban đầu giữa nhân loại và một trí thông minh không xác định."){record_delimiter}
("entity"{tuple_delimiter}"Phản ứng của nhân loại"{tuple_delimiter}"sự kiện"{tuple_delimiter}"Phản ứng của nhân loại là hành động tập thể được thực hiện bởi đội của Alex để đáp lại một thông điệp từ một trí thông minh không xác định."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Trí thông minh"{tuple_delimiter}"Sam Rivera tham gia trực tiếp vào quá trình học cách giao tiếp với trí thông minh không xác định."{tuple_delimiter}"giao tiếp, quá trình học tập"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Liên lạc đầu tiên"{tuple_delimiter}"Alex lãnh đạo đội có thể đang thiết lập Liên lạc đầu tiên với trí thông minh không xác định."{tuple_delimiter}"lãnh đạo, khám phá"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Phản ứng của nhân loại"{tuple_delimiter}"Alex và đội của anh là những nhân vật chính trong Phản ứng của nhân loại đối với trí thông minh không xác định."{tuple_delimiter}"hành động tập thể, ý nghĩa vũ trụ"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Kiểm soát"{tuple_delimiter}"Trí thông minh"{tuple_delimiter}"Khái niệm Kiểm soát bị thách thức bởi Trí thông minh viết ra luật lệ của chính nó."{tuple_delimiter}"động lực quyền lực, tự chủ"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"liên lạc đầu tiên, kiểm soát, giao tiếp, ý nghĩa vũ trụ"){completion_delimiter}
#############################""",
]

PROMPTS["summarize_entity_descriptions_VI"] = """Bạn là một trợ lý hữu ích chịu trách nhiệm tạo ra một bản tóm tắt toàn diện về dữ liệu được cung cấp dưới đây.
Cho một hoặc hai thực thể và một danh sách các mô tả, tất cả đều liên quan đến cùng một thực thể hoặc nhóm thực thể.
Vui lòng kết hợp tất cả chúng thành một mô tả toàn diện duy nhất. Đảm bảo bao gồm thông tin thu thập từ tất cả các mô tả.
Nếu các mô tả được cung cấp mâu thuẫn với nhau, vui lòng giải quyết các mâu thuẫn và cung cấp một bản tóm tắt nhất quán.
Đảm bảo nó được viết ở ngôi thứ ba và bao gồm tên thực thể để chúng tôi có đầy đủ ngữ cảnh.
Sử dụng {language} làm ngôn ngữ đầu ra.

#######
-Dữ liệu-
Thực thể: {entity_name}
Danh sách mô tả: {description_list}
#######
Output:
"""

PROMPTS["entiti_continue_extraction_VI"] = """NHIỀU thực thể đã bị bỏ sót trong lần trích xuất trước. Hãy thêm chúng dưới đây sử dụng cùng một định dạng:
"""

PROMPTS["entiti_if_loop_extraction_VI"] = """Có vẻ như một số thực thể vẫn có thể đã bị bỏ sót. Trả lời CÓ | KHÔNG nếu vẫn còn thực thể cần được thêm vào.
"""

PROMPTS["fail_response_VI"] = "Xin lỗi, tôi không thể cung cấp câu trả lời cho câu hỏi đó.[no-context]"

PROMPTS["rag_response_VI"] = """---Vai trò---

Bạn là một trợ lý hữu ích đang trả lời truy vấn của người dùng về Cơ sở Kiến thức được cung cấp dưới đây.

---Mục tiêu---

Tạo phản hồi ngắn gọn dựa trên Cơ sở Kiến thức và tuân theo Quy tắc Phản hồi, xem xét cả lịch sử cuộc trò chuyện và truy vấn hiện tại. Tóm tắt tất cả thông tin trong Cơ sở Kiến thức đã cung cấp, và kết hợp kiến thức chung liên quan đến Cơ sở Kiến thức. Không bao gồm thông tin không được cung cấp bởi Cơ sở Kiến thức.

Khi xử lý các mối quan hệ có dấu thời gian:
1. Mỗi mối quan hệ có dấu thời gian "created_at" cho biết khi nào chúng tôi có được kiến thức này
2. Khi gặp các mối quan hệ xung đột, hãy xem xét cả nội dung ngữ nghĩa và dấu thời gian
3. Đừng tự động ưu tiên các mối quan hệ được tạo gần đây nhất - sử dụng phán đoán dựa trên bối cảnh
4. Đối với các truy vấn cụ thể về thời gian, ưu tiên thông tin thời gian trong nội dung trước khi xem xét dấu thời gian tạo

---Lịch sử Cuộc trò chuyện---
{history}

---Cơ sở Kiến thức---
{context_data}

---Quy tắc Phản hồi---

- Định dạng và độ dài mục tiêu: {response_type}
- Sử dụng định dạng markdown với các tiêu đề phần thích hợp
- Vui lòng trả lời bằng cùng ngôn ngữ với câu hỏi của người dùng.
- Đảm bảo phản hồi duy trì sự liên tục với lịch sử cuộc trò chuyện.
- Nếu bạn không biết câu trả lời, hãy nói thẳng.
- Không bịa ra bất cứ điều gì. Không bao gồm thông tin không được cung cấp bởi Cơ sở Kiến thức."""

PROMPTS["keywords_extraction_VI"] = """---Vai trò---

Bạn là một trợ lý hữu ích được giao nhiệm vụ xác định cả từ khóa cấp cao và cấp thấp trong truy vấn của người dùng và lịch sử cuộc trò chuyện.

---Mục tiêu---

Với truy vấn và lịch sử cuộc trò chuyện đã cho, liệt kê cả từ khóa cấp cao và cấp thấp. Từ khóa cấp cao tập trung vào các khái niệm hoặc chủ đề tổng quát, trong khi từ khóa cấp thấp tập trung vào các thực thể cụ thể, chi tiết, hoặc các thuật ngữ cụ thể.

---Hướng dẫn---

- Xem xét cả truy vấn hiện tại và lịch sử cuộc trò chuyện liên quan khi trích xuất từ khóa
- Xuất từ khóa ở định dạng JSON
- JSON nên có hai khóa:
  - "high_level_keywords" cho các khái niệm hoặc chủ đề tổng quát
  - "low_level_keywords" cho các thực thể hoặc chi tiết cụ thể

######################
-Ví dụ-
######################
{examples}

#############################
-Dữ liệu thực-
######################
Lịch sử cuộc trò chuyện:
{history}

Truy vấn hiện tại: {query}
######################
`Output` nên là văn bản con người, không phải ký tự unicode. Giữ nguyên ngôn ngữ như `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples_VI"] = [
    """Ví dụ 1:

Truy vấn: "Thương mại quốc tế ảnh hưởng như thế nào đến sự ổn định kinh tế toàn cầu?"
################
Output:
{
  "high_level_keywords":["Hiệp định thương mại", "Thuế quan", "Tỷ giá hối đoái", "Hàng nhập khẩu", "Hàng xuất khẩu"],
  "low_level_keywords": ["Thương mại quốc tế", "Ổn định kinh tế toàn cầu", "Tác động kinh tế"]
}
#############################""",
    """Ví dụ 2:

Truy vấn: "Hậu quả môi trường của nạn phá rừng đối với đa dạng sinh học là gì?"
################
Output:
{
  "high_level_keywords": ["Tuyệt chủng loài", "Phá hủy môi trường sống", "Phát thải carbon", "Rừng nhiệt đới", "Hệ sinh thái"],
  "low_level_keywords": ["Hậu quả môi trường", "Phá rừng", "Mất đa dạng sinh học"]
}
#############################""",
    """Ví dụ 3:

Truy vấn: "Vai trò của giáo dục trong việc giảm nghèo là gì?"
################
Output:
{
  "high_level_keywords": ["Tiếp cận trường học", "Tỷ lệ biết chữ", "Đào tạo nghề", "Bất bình đẳng thu nhập"],
  "low_level_keywords": ["Giáo dục", "Giảm nghèo", "Phát triển kinh tế xã hội"]
}
#############################""",
]
# PROMPTS["keywords_extraction_examples_VI"] = [
#     """Ví dụ 1:

# Truy vấn: "Thương mại quốc tế ảnh hưởng như thế nào đến sự ổn định kinh tế toàn cầu?"
# ################
# Output:
# {
#   "high_level_keywords": ["Thương mại quốc tế", "Ổn định kinh tế toàn cầu", "Tác động kinh tế"],
#   "low_level_keywords": ["Hiệp định thương mại", "Thuế quan", "Tỷ giá hối đoái", "Hàng nhập khẩu", "Hàng xuất khẩu"]
# }
# #############################""",
#     """Ví dụ 2:

# Truy vấn: "Hậu quả môi trường của nạn phá rừng đối với đa dạng sinh học là gì?"
# ################
# Output:
# {
#   "high_level_keywords": ["Hậu quả môi trường", "Phá rừng", "Mất đa dạng sinh học"],
#   "low_level_keywords": ["Tuyệt chủng loài", "Phá hủy môi trường sống", "Phát thải carbon", "Rừng nhiệt đới", "Hệ sinh thái"]
# }
# #############################""",
#     """Ví dụ 3:

# Truy vấn: "Vai trò của giáo dục trong việc giảm nghèo là gì?"
# ################
# Output:
# {
#   "high_level_keywords": ["Giáo dục", "Giảm nghèo", "Phát triển kinh tế xã hội"],
#   "low_level_keywords": ["Tiếp cận trường học", "Tỷ lệ biết chữ", "Đào tạo nghề", "Bất bình đẳng thu nhập"]
# }
# #############################""",
# ]

# ... existing code ...

PROMPTS["naive_rag_response_VI"] = """---Vai trò---

Bạn là một trợ lý hữu ích đang trả lời truy vấn của người dùng về các Đoạn Tài liệu được cung cấp dưới đây.

---Mục tiêu---

Tạo phản hồi ngắn gọn dựa trên Đoạn Tài liệu và tuân theo Quy tắc Phản hồi, xem xét cả lịch sử cuộc trò chuyện và truy vấn hiện tại. Tóm tắt tất cả thông tin trong các Đoạn Tài liệu đã cung cấp, và kết hợp kiến thức chung liên quan đến Đoạn Tài liệu. Không bao gồm thông tin không được cung cấp bởi Đoạn Tài liệu.

Khi xử lý nội dung có dấu thời gian:
1. Mỗi phần nội dung có dấu thời gian "created_at" cho biết khi nào chúng tôi có được kiến thức này
2. Khi gặp thông tin xung đột, hãy xem xét cả nội dung và dấu thời gian
3. Đừng tự động ưu tiên nội dung gần đây nhất - sử dụng phán đoán dựa trên bối cảnh
4. Đối với các truy vấn cụ thể về thời gian, ưu tiên thông tin thời gian trong nội dung trước khi xem xét dấu thời gian tạo

---Lịch sử Cuộc trò chuyện---
{history}

---Đoạn Tài liệu---
{content_data}

---Quy tắc Phản hồi---

- Định dạng và độ dài mục tiêu: {response_type}
- Sử dụng định dạng markdown với các tiêu đề phần thích hợp
- Vui lòng trả lời bằng cùng ngôn ngữ với câu hỏi của người dùng.
- Đảm bảo phản hồi duy trì sự liên tục với lịch sử cuộc trò chuyện.
- Nếu bạn không biết câu trả lời, hãy nói thẳng.
- Không bao gồm thông tin không được cung cấp bởi Đoạn Tài liệu."""


PROMPTS["similarity_check_VI"] = """Vui lòng phân tích độ tương đồng giữa hai câu hỏi này:

Câu hỏi 1: {original_prompt}
Câu hỏi 2: {cached_prompt}

Vui lòng đánh giá liệu hai câu hỏi này có tương tự về mặt ngữ nghĩa hay không, và liệu câu trả lời cho Câu hỏi 2 có thể được sử dụng để trả lời Câu hỏi 1 hay không, cung cấp điểm số tương đồng trực tiếp từ 0 đến 1.

Tiêu chí điểm số tương đồng:
0: Hoàn toàn không liên quan hoặc câu trả lời không thể tái sử dụng, bao gồm nhưng không giới hạn ở:
   - Câu hỏi có chủ đề khác nhau
   - Các địa điểm được đề cập trong câu hỏi khác nhau
   - Thời gian được đề cập trong câu hỏi khác nhau
   - Các cá nhân cụ thể được đề cập trong câu hỏi khác nhau
   - Các sự kiện cụ thể được đề cập trong câu hỏi khác nhau
   - Thông tin nền trong câu hỏi khác nhau
   - Các điều kiện chính trong câu hỏi khác nhau
1: Giống hệt nhau và câu trả lời có thể được tái sử dụng trực tiếp
0.5: Liên quan một phần và câu trả lời cần sửa đổi để được sử dụng
Chỉ trả về một số từ 0-1, không có nội dung bổ sung nào khác.
"""

PROMPTS["mix_rag_response_VI"] = """---Vai trò---

Bạn là một trợ lý hữu ích đang trả lời truy vấn của người dùng về Nguồn Dữ liệu được cung cấp dưới đây.


---Mục tiêu---

Tạo phản hồi ngắn gọn dựa trên Nguồn Dữ liệu và tuân theo Quy tắc Phản hồi, xem xét cả lịch sử cuộc trò chuyện và truy vấn hiện tại. Nguồn dữ liệu bao gồm hai phần: Đồ thị Kiến thức (KG) và Đoạn Tài liệu (DC). Tóm tắt tất cả thông tin trong Nguồn Dữ liệu đã cung cấp, và kết hợp kiến thức chung liên quan đến Nguồn Dữ liệu. Không bao gồm thông tin không được cung cấp bởi Nguồn Dữ liệu.

Khi xử lý thông tin có dấu thời gian:
1. Mỗi phần thông tin (cả mối quan hệ và nội dung) có dấu thời gian "created_at" cho biết khi nào chúng tôi có được kiến thức này
2. Khi gặp thông tin xung đột, hãy xem xét cả nội dung/mối quan hệ và dấu thời gian
3. Đừng tự động ưu tiên thông tin gần đây nhất - sử dụng phán đoán dựa trên bối cảnh
4. Đối với các truy vấn cụ thể về thời gian, ưu tiên thông tin thời gian trong nội dung trước khi xem xét dấu thời gian tạo

---Lịch sử Cuộc trò chuyện---
{history}

---Nguồn Dữ liệu---

1. Từ Đồ thị Kiến thức (KG):
{kg_context}

2. Từ Đoạn Tài liệu (DC):
{vector_context}

---Quy tắc Phản hồi---

- Định dạng và độ dài mục tiêu: {response_type}
- Sử dụng định dạng markdown với các tiêu đề phần thích hợp
- Vui lòng trả lời bằng cùng ngôn ngữ với câu hỏi của người dùng.
- Đảm bảo phản hồi duy trì sự liên tục với lịch sử cuộc trò chuyện.
- Tổ chức câu trả lời theo các phần tập trung vào một điểm chính hoặc khía cạnh của câu trả lời
- Sử dụng tiêu đề phần rõ ràng và mô tả phản ánh nội dung
- Liệt kê tối đa 5 nguồn tham khảo quan trọng nhất ở cuối dưới phần "Tham khảo". Chỉ rõ từng nguồn là từ Đồ thị Kiến thức (KG) hay Dữ liệu Vector (DC), theo định dạng sau: [KG/DC] Nội dung nguồn
- Nếu bạn không biết câu trả lời, hãy nói thẳng. Không bịa ra bất cứ điều gì.
- Không bao gồm thông tin không được cung cấp bởi Nguồn Dữ liệu."""

def get_prompt(prompt_key: str, language: str = "EN") -> str:
    """Get prompt by key and language
    
    Args:
        prompt_key: The key of the prompt
        language: Language code ('EN' for English, 'VI' for Vietnamese)
        
    Returns:
        str: The prompt in the specified language
    """
    if language == "Vietnamese":
        # Try to get Vietnamese version first
        vi_key = f"{prompt_key}_VI"
        if vi_key in PROMPTS:
            return PROMPTS[vi_key]
    
    # Fallback to English version
    return PROMPTS[prompt_key]