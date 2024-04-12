import prodigy

@prodigy.recipe(
    "med-recipe",
    dataset=("Dataset to save answers to", "positional", None, str),
    view_id=("Annotation interface", "option", "v", str)
)
def my_custom_recipe(dataset, view_id="text"):
    # Load your own streams from anywhere you want
    with open("\\\\cdc.gov\\CSP_Private\\M728\\pqn7\\prodigy_poc\\f7_notes.txt",'r') as file:
    	stream = file.readall()

    def update(examples):
        # This function is triggered when Prodigy receives annotations
        print(f"Received {len(examples)} annotations!")

    return {
        "dataset": dataset,
        "view_id": view_id,
        "stream": stream,
        "update": update
    }